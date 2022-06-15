import logging
import random

import torch
from torch.utils.data.dataloader import DataLoader

import pydiffvg

from .models.my_discriminator import MyDiscriminator
from .models.my_gen_fixed_6figs32_good import MyGeneratorFixedSixFigs32Good
from .models.my_generator import MyGenerator
from .models.generator import Generator
from .models.discriminator import Discriminator

from utils.noise import get_noise
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.plotting import plot_losses, plot_grad_flow, plot_real_fake_covers
from .models.my_generator_circle_paths import MyGeneratorCircled
from .models.my_generator_fixed_figure import MyGeneratorFixedFigure
from .models.my_generator_fixed_six_figs import MyGeneratorFixedSixFigs
from .models.my_generator_fixed_six_figs_backup import MyGeneratorFixedSixFigs32
from .models.my_generator_oval_paths_multi import MyGeneratorOvaledMulti
from .models.generator_multi_shape import GeneratorMultiShape
from .models.my_generator_fixed_multi_circle import MyGeneratorFixedMultiCircle
from .models.my_generator_rand_figure import MyGeneratorRandFigure
from .models.my_generator_three_figs import MyGeneratorFixedThreeFigs32

logger = logging.getLogger("trainer")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

# generator_type = Generator
# generator_type = MyGenerator
# generator_type = MyGeneratorCircled
# generator_type = MyGeneratorFixedMultiCircle
# generator_type = MyGeneratorFixedFigure
# generator_type = MyGeneratorRandFigure
generator_type = MyGeneratorFixedSixFigs
generator_type = MyGeneratorFixedSixFigs32
generator_type = MyGeneratorFixedThreeFigs32
generator_type = MyGeneratorFixedSixFigs32Good
# generator_type = MyGeneratorOvaledMulti
# generator_type = GeneratorMultiShape

discriminator_type = Discriminator


# discriminator_type = MyDiscriminator

def make_gan_models(z_dim: int, audio_embedding_dim: int, img_shape: (int, int, int), has_emotions: bool,
                    num_gen_layers: int, num_disc_conv_layers: int, num_disc_linear_layers: int,
                    path_count: int, path_segment_count: int, max_stroke_width: float, disc_slices: int,
                    device: torch.device) -> (Generator, Discriminator):
    canvas_size = img_shape[1]

    gen = generator_type(
        z_dim=z_dim,
        audio_embedding_dim=audio_embedding_dim * disc_slices,
        has_emotions=has_emotions,
        num_layers=num_gen_layers,
        canvas_size=canvas_size,
        path_count=path_count,
        path_segment_count=path_segment_count,
        max_stroke_width=max_stroke_width
    ).to(device)
    disc = discriminator_type(
        canvas_size=canvas_size,
        audio_embedding_dim=audio_embedding_dim * disc_slices,
        has_emotions=has_emotions,
        num_conv_layers=num_disc_conv_layers,
        num_linear_layers=num_disc_linear_layers
    ).to(device)

    return gen, disc


def get_gradient_penalty(disc, real, fake, audio_embedding, emotions):
    # Mix the images together
    epsilon = torch.rand((real.size(0), 1, 1, 1), device=real.device)
    mixed_images = (real * epsilon + fake * (1 - epsilon)).requires_grad_(True)

    # Calculate the critic's scores on the mixed images
    mixed_scores = disc(mixed_images, audio_embedding, emotions)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    penalty = ((gradient_norm - 1) ** 2).mean()
    return penalty


def mismatching_permute(t: torch.Tensor) -> torch.Tensor:
    shift = random.randrange(start=0, stop=len(t))
    return torch.cat((t[shift:], t[:shift]))


def train(dataloader: DataLoader, test_dataloader: DataLoader,
          gen: Generator, disc: Discriminator,
          device: torch.device, training_params: dict,
          cgan_out_name='cgan_out',
          palette_generator=None,
          USE_SHUFFLING=False):
    def generate(z, audio_embedding_disc, emotions):
        if palette_generator is None:
            return gen(z, audio_embedding_disc, emotions)
        return gen(z, audio_embedding_disc, emotions, palette_generator=palette_generator)

    logger.info(f'PyDiffVG uses GPU: {pydiffvg.get_use_gpu()}')
    logger.info(gen)
    logger.info(disc)

    n_epochs = training_params["n_epochs"]
    disc_repeats = training_params["disc_repeats"]

    if "lr" in training_params:
        disc_lr = gen_lr = training_params["lr"]
    else:
        gen_lr = training_params["gen_lr"]
        disc_lr = training_params["disc_lr"]
    z_dim = training_params["z_dim"]
    disc_slices = training_params["disc_slices"]
    checkpoint_root = training_params["checkpoint_root"]
    display_steps = training_params["display_steps"]
    backup_epochs = training_params["backup_epochs"]
    bin_steps = training_params["bin_steps"]
    plot_grad = training_params["plot_grad"]
    c_lambda = 10

    gen_opt = torch.optim.Adam(gen.parameters(), lr=gen_lr, betas=(0.5, 0.9))
    disc_opt = torch.optim.Adam(disc.parameters(), lr=disc_lr, betas=(0.5, 0.9))

    print("Trying to load checkpoint.")
    epochs_done = load_checkpoint(checkpoint_root, cgan_out_name, [gen, disc, gen_opt, disc_opt])
    if epochs_done:
        logger.info(f"Loaded a checkpoint with {epochs_done} epochs done")

    cur_step = 0
    generator_losses = []
    discriminator_losses = []
    shuffle_losses = []
    val_metrics = []

    disc_repeat_cnt = 0
    mean_iteration_disc_loss, mean_shuffle_disc_loss = 0, 0
    for epoch in range(epochs_done + 1, n_epochs + epochs_done + 1):
        for batch in dataloader:
            torch.cuda.empty_cache()
            if len(batch) == 3:
                audio_embedding, real_cover_tensor, emotions = batch
                emotions = emotions.to(device)
            else:
                audio_embedding, real_cover_tensor = batch
                emotions = None
            cur_batch_size = len(audio_embedding)
            audio_embedding = audio_embedding.float().to(device)
            audio_embedding_disc = audio_embedding[:, :disc_slices].reshape(cur_batch_size, -1)
            real_cover_tensor = real_cover_tensor.to(device)
            shuffle_cover_tensor = mismatching_permute(real_cover_tensor)

            def get_fake_pred(should_detach: bool):
                # Get noise corresponding to the current batch_size
                z = get_noise(cur_batch_size, z_dim, device=device)

                if should_detach:
                    with torch.no_grad():
                        fake_covers = generate(z, audio_embedding_disc, emotions)
                        fig_params = None
                else:
                    # fake_covers, fig_params = gen(z, audio_embedding_disc, emotions, return_figure_params=True)
                    fake_covers = generate(z, audio_embedding_disc, emotions)
                    fig_params = None
                # Make sure that enough shapes were generated
                assert len(fake_covers) == len(real_cover_tensor)

                fake_pred = disc(fake_covers, audio_embedding_disc, emotions)

                assert len(fake_pred) == len(fake_covers)
                return fake_covers, fake_pred, fig_params

            # ### Update discriminator with fakes ###
            # Zero out the discriminator gradients
            disc_opt.zero_grad()

            fake_cover_tensor, disc_fake_pred, fig_params = get_fake_pred(should_detach=True)
            disc_real_pred = disc(real_cover_tensor, audio_embedding_disc, emotions)

            # Make sure that enough predictions were made
            assert len(disc_real_pred) == len(real_cover_tensor)
            # Shapes must match
            assert tuple(disc_fake_pred.shape) == tuple(disc_real_pred.shape)

            gp = get_gradient_penalty(disc, real_cover_tensor.data, fake_cover_tensor.data,
                                      audio_embedding_disc, emotions)
            disc_loss = disc_fake_pred.mean() - disc_real_pred.mean() + gp * c_lambda
            # Keep track of the average critic loss in this batch
            mean_iteration_disc_loss += disc_loss.item() / disc_repeats
            disc_repeat_cnt += 1
            # Update gradients
            disc_loss.backward()
            if plot_grad:
                plot_grad_flow(disc.named_parameters(), "discriminator (fakes)", epoch=epoch, cur_step=cur_step)
            # Update optimizer
            disc_opt.step()

            if USE_SHUFFLING:
                # ### Update discriminator with matching and shuffled cover-embedding pairs ###
                disc_opt.zero_grad()

                disc_real_pred = disc(real_cover_tensor, audio_embedding_disc, emotions)
                disc_shuffle_pred = disc(shuffle_cover_tensor, audio_embedding_disc, emotions)
                disc_loss = disc_shuffle_pred.mean() - disc_real_pred.mean()

                mean_shuffle_disc_loss += disc_loss.item() / disc_repeats

                # Update gradients
                disc_loss.backward()
                if plot_grad:
                    plot_grad_flow(disc.named_parameters(), "discriminator (shuffled pairs)", epoch=epoch,
                                   cur_step=cur_step)
                disc_opt.step()

            if disc_repeat_cnt == disc_repeats:
                # Keep track of the average discriminator loss
                discriminator_losses.append(mean_iteration_disc_loss)
                shuffle_losses.append(mean_shuffle_disc_loss)
                disc_repeat_cnt = 0
                mean_iteration_disc_loss, mean_shuffle_disc_loss = 0, 0

                # ### Update generator ###
                # Zero out the generator gradients
                gen_opt.zero_grad()

                # Getting fake shapes, same as in the loop above
                fake_cover_tensor, disc_fake_pred, fig_params = get_fake_pred(should_detach=False)

                # p_dist = torch.nn.PairwiseDistance()
                # dist_loss_sum = torch.tensor(0.0).to(fake_cover_tensor.device)
                # for figs in fig_params:
                #     cur_batch_dist_loss = torch.tensor(0.0).to(fake_cover_tensor.device)
                #     for ind_x, x in enumerate(figs):
                #         for ind_y in range(ind_x + 1, len(figs)):
                #             a = x["center_point"]
                #             b = figs[ind_y]["center_point"]
                #             cur_batch_dist_loss += 50000 / (p_dist(a, b)) ** 3
                #     dist_loss_sum += cur_batch_dist_loss / len(figs)
                #
                # dist_loss_sum = dist_loss_sum / len(fig_params)
                # dist_loss_sum.backward()
                # gen_loss = -disc_fake_pred.mean() + dist_loss_sum
                gen_loss = -disc_fake_pred.mean()
                gen_loss.backward()
                if plot_grad:
                    plot_grad_flow(gen.named_parameters(), "generator", epoch=epoch, cur_step=cur_step)

                # Update the weights
                gen_opt.step()

                # Keep track of the generator losses
                generator_losses.append(gen_loss.item())

            plot_losses(epoch, cur_step, display_steps, bin_steps, [
                ("Generator", generator_losses),
                ("Discriminator Adversarial", discriminator_losses),
                ("Discriminator Mismatches", shuffle_losses)
            ])
            if cur_step % display_steps == 0:
                plot_real_fake_covers(real_cover_tensor, fake_cover_tensor, disc_real_pred, disc_fake_pred, epoch=epoch,
                                      cur_step=cur_step)
                save_checkpoint(checkpoint_root, cgan_out_name, epoch, 0, [gen, disc, gen_opt, disc_opt])
            cur_step += 1

        if test_dataloader is not None:
            gen.eval()
            disc.eval()
            mean_val_pred = 0
            for batch in test_dataloader:
                if len(batch) == 3:
                    audio_embedding, _real_cover_tensor, emotions = batch
                    emotions = emotions.to(device)
                else:
                    audio_embedding, _real_cover_tensor = batch
                    emotions = None
                cur_batch_size = len(audio_embedding)
                audio_embedding = audio_embedding.float().to(device)
                audio_embedding_disc = audio_embedding[:, :disc_slices].reshape(cur_batch_size, -1)

                noise = get_noise(cur_batch_size, z_dim, device=device)

                fake_cover_tensor = generate(noise, audio_embedding_disc, emotions)
                mean_val_pred += disc(fake_cover_tensor, audio_embedding_disc, emotions).mean().item()
            mean_val_pred /= len(test_dataloader)
            val_metrics.append(mean_val_pred)
            if epoch % backup_epochs == 0:
                plot_losses(epoch, 1, 1, 1, [("Val pred metric", val_metrics)], epoch=epoch, cur_step=cur_step)
            gen.train()
            disc.train()

        save_checkpoint(checkpoint_root, cgan_out_name, epoch, backup_epochs, [gen, disc, gen_opt, disc_opt])
