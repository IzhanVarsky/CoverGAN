#!/usr/bin/env python3
# coding: utf-8
import argparse
import logging
import os

import torch
from torch.utils.data.dataloader import DataLoader

from colorer.models.colorer import Colorer
from colorer.models.colorer_dropout import Colorer2
from colorer.music_palette_dataset import MusicPaletteDataset

logger = logging.getLogger("colorer_train")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

colorer_type = Colorer
colorer_type = Colorer2


def get_train_data(checkpoint_dir: str, audio_dir: str, cover_dir: str, emotion_file: str,
                   batch_size: int, is_for_train: bool = True) -> \
        (DataLoader, int, (int, int, int), bool):
    dataset = MusicPaletteDataset("cgan_out_dataset", checkpoint_dir,
                                  audio_dir, cover_dir, emotion_file, is_for_train=is_for_train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    music_tensor, palette_tensor = dataset[0][:2]
    audio_embedding_dim = music_tensor.shape[1]
    palette_shape = palette_tensor.shape
    has_emotions = dataset.has_emotions()

    return dataloader, audio_embedding_dim, palette_shape, has_emotions


def file_in_folder(dir, file):
    if file is None:
        return None
    return f"{dir}/{file}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", help="Directory with all folders for training", type=str, default=".")
    parser.add_argument("--colors_count", help="Count of colors to predict", type=int, default=6)
    parser.add_argument("--plots", help="Directory where save plots while training", type=str, default="plots")
    parser.add_argument("--audio", help="Directory with the music files", type=str, default="audio")
    parser.add_argument("--covers", help="Directory with the cover images", type=str, default="clean_covers")
    parser.add_argument("--emotions", help="File with emotion markup for train dataset", type=str, default=None)
    parser.add_argument("--test_set", help="Directory with test music files", type=str, default=None)
    parser.add_argument("--test_emotions", help="File with emotion markup for test dataset", type=str, default=None)
    parser.add_argument("--checkpoint_root", help="Checkpoint location", type=str, default="checkpoint")
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.0005)
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
    z_dim = 32  # Dimension of the noise vector

    disc_slices = 6

    # Plot properties
    bin_steps = 20  # How many steps to aggregate with mean for each plot point

    logger.info("--- Starting out_main ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(file_in_folder(args.train_dir, args.checkpoint_root), exist_ok=True)

    train_dataloader, audio_embedding_dim, img_shape, has_emotions = get_train_data(
        file_in_folder(args.train_dir, args.checkpoint_root),
        file_in_folder(args.train_dir, args.audio),
        file_in_folder(args.train_dir, args.covers),
        file_in_folder(args.train_dir, args.emotions),
        args.batch_size, is_for_train=True
    )

    test_dataloader, audio_embedding_dim, img_shape, has_emotions = get_train_data(
        file_in_folder(args.train_dir, args.checkpoint_root),
        file_in_folder(args.train_dir, args.audio),
        file_in_folder(args.train_dir, args.covers),
        file_in_folder(args.train_dir, args.emotions),
        args.batch_size, is_for_train=False
    )

    logger.info("--- Colorer training ---")
    gen = colorer_type(
        z_dim=z_dim,
        audio_embedding_dim=audio_embedding_dim * disc_slices,
        has_emotions=has_emotions,
        num_layers=num_gen_layers,
        colors_count=args.colors_count,
    ).to(device)
    GAN_MODEL = True
    training_params = {
        # Common
        "display_steps": args.display_steps,
        "backup_epochs": args.backup_epochs,
        "bin_steps": bin_steps,
        "z_dim": z_dim,
        "disc_slices": disc_slices,
        "checkpoint_root": file_in_folder(args.train_dir, args.checkpoint_root),
        # (W)GAN-specific
        "n_epochs": args.epochs,
        "lr": args.lr,
        "disc_repeats": args.disc_repeats,
        "plot_grad": args.plot_grad,
    }
    if not GAN_MODEL:
        from colorer.train import train
        train(train_dataloader, gen, device, training_params, test_dataloader)
    else:
        from colorer.models.gan_colorer import ColorerDiscriminator, train
        num_disc_layers = 2
        disc = ColorerDiscriminator(audio_embedding_dim=audio_embedding_dim * disc_slices,
                                    has_emotions=has_emotions,
                                    num_layers=num_disc_layers).to(device)
        train(train_dataloader, test_dataloader, gen, disc, device, training_params)


if __name__ == '__main__':
    main()
