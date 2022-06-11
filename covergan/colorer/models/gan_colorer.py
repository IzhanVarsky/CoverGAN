import logging
from typing import Optional

import torch
from torch.utils.data import DataLoader

from outer.emotions import Emotion
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.noise import get_noise

logger = logging.getLogger("colorer gan trainer")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class ColorerDiscriminator(torch.nn.Module):
    def __init__(self, audio_embedding_dim: int, has_emotions: bool, num_layers: int):
        super(ColorerDiscriminator, self).__init__()
        colors_count = 12
        self.colors_count = colors_count
        in_features = audio_embedding_dim + colors_count * 3
        if has_emotions:
            in_features += len(Emotion)
        out_features = 1

        feature_step = (in_features - out_features) // num_layers

        layers = []
        for i in range(num_layers - 1):
            out_features = in_features - feature_step
            layers += [
                torch.nn.Linear(in_features=in_features, out_features=out_features),
                torch.nn.LeakyReLU(0.2),
                torch.nn.Dropout2d(0.2)
            ]
            in_features = out_features
        layers += [
            torch.nn.Linear(in_features=in_features, out_features=colors_count),
            torch.nn.Sigmoid()
        ]
        self.adv_layer = torch.nn.Sequential(*layers)

    def forward(self, audio_embedding: torch.Tensor, emotions: Optional[torch.Tensor],
                colors: torch.Tensor) -> torch.Tensor:
        if emotions is None:
            cat = torch.cat((audio_embedding, colors), dim=1)
        else:
            cat = torch.cat((audio_embedding, emotions, colors), dim=1)
        validity = self.adv_layer(cat)
        assert not torch.any(torch.isnan(validity))
        return validity


def train(train_dataloader: DataLoader,
          test_dataloader: DataLoader,
          gen, disc: ColorerDiscriminator,
          device: torch.device,
          training_params: dict):
    n_epochs = training_params["n_epochs"]
    lr = training_params["lr"]
    z_dim = training_params["z_dim"]
    disc_slices = training_params["disc_slices"]
    checkpoint_root = training_params["checkpoint_root"]
    backup_epochs = training_params["backup_epochs"]

    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    disc_opt = torch.optim.Adam(disc.parameters(), lr=lr)
    model_name = f'GAN_colorer_{gen.colors_count}_colors_{train_dataloader.dataset.sorted_color}'
    print("Trying to load checkpoint.")
    epochs_done = load_checkpoint(checkpoint_root, model_name, [gen, disc, gen_opt, disc_opt])
    if epochs_done:
        logger.info(f"Loaded a checkpoint with {epochs_done} epochs done")

    criterion = torch.nn.MSELoss()
    log_interval = 1
    colors_count = gen.colors_count
    disc_getting_colors_count = 1
    for epoch in range(epochs_done + 1, n_epochs + epochs_done + 1):
        gen.train()
        disc.train()
        for batch_idx, batch in enumerate(train_dataloader):
            torch.cuda.empty_cache()
            if len(batch) == 3:
                audio_embedding, real_palette, emotions = batch
                real_palette = real_palette.to(device)
                emotions = emotions.to(device)
            else:
                audio_embedding, real_palette = batch
                real_palette = real_palette.to(device)
                emotions = None
            cur_batch_size = len(audio_embedding)
            audio_embedding = audio_embedding.float().to(device)
            audio_embedding_disc = audio_embedding[:, :disc_slices].reshape(cur_batch_size, -1)

            # Train discriminator
            real_outputs = disc(audio_embedding_disc, emotions, real_palette)
            real_label = torch.ones(real_palette.shape[0], colors_count).to(device)
            z = get_noise(cur_batch_size, z_dim, device=device)
            fake_inputs = gen(z, audio_embedding_disc, emotions)
            fake_outputs = disc(audio_embedding_disc, emotions, fake_inputs)
            fake_label = torch.zeros(fake_inputs.shape[0], colors_count).to(device)
            outputs = torch.cat((real_outputs, fake_outputs), dim=0)
            targets = torch.cat((real_label, fake_label), dim=0)
            D_loss = criterion(outputs, targets)
            disc_opt.zero_grad()
            D_loss.backward()
            disc_opt.step()
            # Train generator
            z = get_noise(cur_batch_size, z_dim, device=device)
            fake_inputs = gen(z, audio_embedding_disc, emotions)
            fake_outputs = disc(audio_embedding_disc, emotions, fake_inputs)
            fake_targets = torch.ones(fake_inputs.shape[0], colors_count).to(device)
            G_loss = criterion(fake_outputs, fake_targets)
            gen_opt.zero_grad()
            G_loss.backward()
            gen_opt.step()
            if batch_idx % 10 == 0 or batch_idx == len(train_dataloader):
                print('Epoch {} Iteration {}: discriminator_loss {:.3f} generator_loss {:.3f}'
                      .format(epoch, batch_idx, D_loss.item(), G_loss.item()))
        save_checkpoint(checkpoint_root, model_name, epoch, backup_epochs, [gen, disc, gen_opt, disc_opt])

        if epoch == epochs_done + 1 or epoch % log_interval == 0:
            gen.eval()
            disc.eval()
            running_G_test_loss = 0.0
            running_D_test_loss = 0.0
            for batch_idx, batch in enumerate(test_dataloader):
                torch.cuda.empty_cache()
                if len(batch) == 3:
                    audio_embedding, real_palette, emotions = batch
                    real_palette = real_palette.to(device)
                    emotions = emotions.to(device)
                else:
                    audio_embedding, real_palette = batch
                    real_palette = real_palette.to(device)
                    emotions = None
                cur_batch_size = len(audio_embedding)
                audio_embedding = audio_embedding.float().to(device)
                audio_embedding_disc = audio_embedding[:, :disc_slices].reshape(cur_batch_size, -1)

                z = get_noise(cur_batch_size, z_dim, device=device)
                net_out = gen(z, audio_embedding_disc, emotions)
                loss = criterion(net_out * 255, real_palette)
                running_G_test_loss += loss

                real_outputs = disc(audio_embedding_disc, emotions, real_palette)
                real_label = torch.ones(real_palette.shape[0], colors_count).to(device)
                fake_inputs = net_out
                fake_outputs = disc(audio_embedding_disc, emotions, fake_inputs)
                fake_label = torch.zeros(fake_inputs.shape[0], colors_count).to(device)
                outputs = torch.cat((real_outputs, fake_outputs), dim=0)
                targets = torch.cat((real_label, fake_label), dim=0)
                D_loss = criterion(outputs, targets)
                running_D_test_loss += D_loss

            avg_G_test_loss = running_G_test_loss / (batch_idx + 1)
            avg_D_test_loss = running_D_test_loss / (batch_idx + 1)
            print('Test LOSS: gen {}, disc {}'.format(avg_G_test_loss, avg_D_test_loss))
