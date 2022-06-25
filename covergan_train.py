#!/usr/bin/env python3
# coding: utf-8
import argparse
import logging
import os

import torch
from torch.utils.data.dataloader import DataLoader

from colorer.test_model import get_palette_predictor
from outer.dataset import MusicDataset

from outer.train import make_gan_models, train

from utils.noise import get_noise
from utils.plotting import plot_real_fake_covers

logger = logging.getLogger("out_main")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def get_train_data(checkpoint_dir: str, audio_dir: str, cover_dir: str, emotion_file: str,
                   batch_size: int, canvas_size: int,
                   augment_dataset: bool) -> (DataLoader, int, (int, int, int), bool):
    dataset = MusicDataset("cgan_out_dataset", checkpoint_dir,
                           audio_dir, cover_dir, emotion_file,
                           canvas_size, augment_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    music_tensor, cover_tensor = dataset[0][:2]
    audio_embedding_dim = music_tensor.shape[1]
    img_shape = cover_tensor.shape
    has_emotions = dataset.has_emotions()

    return dataloader, audio_embedding_dim, img_shape, has_emotions


def get_test_data(checkpoint_dir: str, test_set_dir: str, test_emotion_file: str,
                  batch_size: int, canvas_size: int) -> (DataLoader, int, (int, int, int), bool):
    test_dataset = MusicDataset("cgan_out_test_dataset", checkpoint_dir,
                                test_set_dir, test_set_dir,
                                test_emotion_file, canvas_size)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return test_dataloader


def demo_samples(gen, dataloader: DataLoader, z_dim: int, disc_slices: int, device: torch.device,
                 palette_generator=None):
    def generate(z, audio_embedding_disc, emotions):
        if palette_generator is None:
            return gen(z, audio_embedding_disc, emotions)
        return gen(z, audio_embedding_disc, emotions, palette_generator=palette_generator)

    gen.eval()

    sample_count = 5  # max covers to draw

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 2:
                audio_embedding, real_cover_tensor = batch
                emotions = None
            else:
                audio_embedding, real_cover_tensor, emotions = batch
                emotions = emotions[:sample_count].to(device)
            sample_count = min(sample_count, len(audio_embedding))
            audio_embedding = audio_embedding[:sample_count].float().to(device)
            audio_embedding = audio_embedding[:, :disc_slices].reshape(sample_count, -1)
            real_cover_tensor = real_cover_tensor[:sample_count].to(device)

            noise = get_noise(sample_count, z_dim, device=device)
            fake_cover_tensor = generate(noise, audio_embedding, emotions)

            plot_real_fake_covers(real_cover_tensor, fake_cover_tensor)
            break  # we only want one batch

    gen.train()


def display_dataset_objects(dataloader: DataLoader, disc_slices: int):
    from sklearn.decomposition import PCA
    import matplotlib.pyplot as plt

    all_audio_embeddings = []
    for batch in dataloader:
        audio_embedding = batch[0]
        batch_size = len(audio_embedding)
        audio_embedding = audio_embedding[:, :disc_slices].reshape(batch_size, -1)
        all_audio_embeddings.append(audio_embedding.float())
    all_audio_embeddings = torch.cat(all_audio_embeddings, dim=0).cpu().numpy()

    pca = PCA(n_components=2)
    xy = pca.fit_transform(all_audio_embeddings)
    x = list(xy[:, 0])
    y = list(xy[:, 1])

    plt.scatter(x, y, s=1)
    plt.title("Training tracks")
    plt.show()


def file_in_folder(dir, file):
    if file is None:
        return None
    return f"{dir}/{file}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", help="Directory with all folders for training", type=str, default=".")
    parser.add_argument("--plots", help="Directory where save plots while training", type=str, default="plots")
    parser.add_argument("--audio", help="Directory with the music files", type=str, default="audio")
    parser.add_argument("--covers", help="Directory with the cover images", type=str, default="clean_covers")
    parser.add_argument("--emotions", help="File with emotion markup for train dataset", type=str, default=None)
    parser.add_argument("--test_set", help="Directory with test music files", type=str, default=None)
    parser.add_argument("--test_emotions", help="File with emotion markup for test dataset", type=str, default=None)
    parser.add_argument("--checkpoint_root", help="Checkpoint location", type=str, default="checkpoint")
    parser.add_argument("--augment_dataset", help="Whether to augment the dataset", default=False, action="store_true")
    parser.add_argument("--gen_lr", help="Generator learning rate", type=float, default=0.0005)
    parser.add_argument("--disc_lr", help="Discriminator learning rate", type=float, default=0.0005)
    parser.add_argument("--disc_repeats", help="Discriminator runs per iteration", type=int, default=5)
    parser.add_argument("--epochs", help="Number of epochs to train for", type=int, default=8000)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("--canvas_size", help="Image canvas size for learning", type=int, default=128)
    parser.add_argument("--display_steps", help="How often to plot the samples", type=int, default=500)
    parser.add_argument("--backup_epochs", help="How often to backup checkpoints", type=int, default=600)
    parser.add_argument("--plot_grad", help="Whether to plot the gradients", default=False, action="store_true")
    args = parser.parse_args()
    print(args)

    # Network properties
    num_gen_layers = 5
    num_disc_conv_layers = 3
    num_disc_linear_layers = 2
    z_dim = 32  # Dimension of the noise vector
    # z_dim = 512  # Dimension of the noise vector

    # Painter properties
    path_count = 3
    path_segment_count = 4
    disc_slices = 6
    max_stroke_width = 0.01  # relative to the canvas size

    # Plot properties
    bin_steps = 20  # How many steps to aggregate with mean for each plot point

    logger.info("--- Starting out_main ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(file_in_folder(args.train_dir, args.checkpoint_root), exist_ok=True)

    dataloader, audio_embedding_dim, img_shape, has_emotions = get_train_data(
        file_in_folder(args.train_dir, args.checkpoint_root),
        file_in_folder(args.train_dir, args.audio),
        file_in_folder(args.train_dir, args.covers),
        file_in_folder(args.train_dir, args.emotions),
        args.batch_size, args.canvas_size, args.augment_dataset
    )
    logger.plots_dir = file_in_folder(args.train_dir, args.plots)
    os.makedirs(logger.plots_dir, exist_ok=True)
    # display_dataset_objects(dataloader, disc_slices)
    if args.test_set is None:
        test_dataloader = None
    else:
        test_dataloader = get_test_data(
            file_in_folder(args.train_dir, args.checkpoint_root),
            file_in_folder(args.train_dir, args.test_set),
            file_in_folder(args.train_dir, args.test_emotions),
            args.batch_size, args.canvas_size
        )

    logger.info("--- CoverGAN training ---")
    gen, disc = make_gan_models(
        z_dim=z_dim,
        audio_embedding_dim=audio_embedding_dim,
        img_shape=img_shape,
        has_emotions=has_emotions,
        num_gen_layers=num_gen_layers,
        num_disc_conv_layers=num_disc_conv_layers,
        num_disc_linear_layers=num_disc_linear_layers,
        path_count=path_count,
        path_segment_count=path_segment_count,
        max_stroke_width=max_stroke_width,
        disc_slices=disc_slices,
        device=device
    )
    palette_generator = get_palette_predictor(device)

    demo_samples(gen, dataloader, z_dim, disc_slices, device, palette_generator=palette_generator)

    train(dataloader, test_dataloader, gen, disc, device, {
        # Common
        "display_steps": args.display_steps,
        "backup_epochs": args.backup_epochs,
        "bin_steps": bin_steps,
        "z_dim": z_dim,
        "disc_slices": disc_slices,
        "checkpoint_root": file_in_folder(args.train_dir, args.checkpoint_root),
        # (W)GAN-specific
        "n_epochs": args.epochs,
        "gen_lr": args.gen_lr,
        "disc_lr": args.disc_lr,
        "disc_repeats": args.disc_repeats,
        "plot_grad": args.plot_grad,
    }, cgan_out_name="cgan_6figs_32noise_separated_palette_tanh_betas", palette_generator=palette_generator,
          USE_SHUFFLING=True)

    logger.info("--- CoverGAN sample demo ---")
    demo_samples(gen, dataloader, z_dim, disc_slices, device, palette_generator=palette_generator)


if __name__ == '__main__':
    main()
