import logging

import torch
from torch.utils.data.dataloader import DataLoader

from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.noise import get_noise
from .models.colorer import Colorer

logger = logging.getLogger("trainer")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def weighted_mse_loss(input, target, weight=None):
    if weight is None:
        max_weight = input.size()[1]
        weight = torch.tensor([(max_weight - i // 3) // 3 for i in range(max_weight)]).to(input.device)
        weight = weight.repeat((len(input), 1))
    return (weight * (input - target) ** 2).mean()


def train(train_dataloader: DataLoader, gen: Colorer, device: torch.device, training_params: dict,
          test_dataloader: DataLoader):
    n_epochs = training_params["n_epochs"]
    lr = training_params["lr"]
    z_dim = training_params["z_dim"]
    disc_slices = training_params["disc_slices"]
    checkpoint_root = training_params["checkpoint_root"]
    backup_epochs = training_params["backup_epochs"]

    gen_opt = torch.optim.Adam(gen.parameters(), lr=lr)
    criterion = weighted_mse_loss

    # model_name = f'colorer_{gen.color_type}_{gen.colors_count}_colors'
    model_name = f'colorer_{gen.colors_count}_colors_{train_dataloader.dataset.sorted_color}'
    print("Trying to load checkpoint.")
    epochs_done = load_checkpoint(checkpoint_root, model_name, [gen, gen_opt])
    if epochs_done:
        logger.info(f"Loaded a checkpoint with {epochs_done} epochs done")

    log_interval = 1
    for epoch in range(epochs_done + 1, n_epochs + epochs_done + 1):
        gen.train()
        running_train_loss = 0.0
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

            z = get_noise(cur_batch_size, z_dim, device=device)

            gen_opt.zero_grad()
            net_out = gen(z, audio_embedding_disc, emotions)
            loss = criterion(net_out * 255, real_palette)
            loss.backward()
            gen_opt.step()
            running_train_loss += loss.item()
            # if (batch_idx + 1) % log_interval == 0:
            #     print('Train Epoch: {} [({:.0f}%)] Loss: {:.6f}'.format(
            #         epoch, 100. * batch_idx / len(dataloader), loss.data.item()))
            # save_checkpoint(checkpoint_root, model_name, epoch, backup_epochs, [gen, gen_opt])
        if epoch == epochs_done + 1 or epoch % log_interval == 0:
            print('Train Epoch: {}. Loss: {:.6f}'.format(epoch, loss.item()))

        save_checkpoint(checkpoint_root, model_name, epoch, backup_epochs, [gen, gen_opt])

        avg_train_loss = running_train_loss / (batch_idx + 1)

        if epoch == epochs_done + 1 or epoch % log_interval == 0:
            gen.eval()
            running_test_loss = 0.0
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
                # loss = criterion(net_out * 255, real_palette * 255)
                loss = criterion(net_out * 255, real_palette)
                running_test_loss += loss

            avg_test_loss = running_test_loss / (batch_idx + 1)
            print('LOSS: train {}; valid {}'.format(avg_train_loss, avg_test_loss))
