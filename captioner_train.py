#!/usr/bin/env python3
# coding: utf-8
import argparse
import logging

import torch
from torch.utils.data.dataloader import DataLoader

from captions.dataset import CaptionDataset

from captions.train import make_models, train

logger = logging.getLogger("captioner_main")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def get_train_data(checkpoint_dir: str, original_cover_dir: str, clean_cover_dir: str,
                   batch_size: int, canvas_size: int) -> DataLoader:
    dataset = CaptionDataset(checkpoint_dir, original_cover_dir, clean_cover_dir, canvas_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True
    )

    return dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--original_covers", help="Directory with the original cover images",
                        type=str, default="./original_covers")
    parser.add_argument("--clean_covers", help="Directory with the cover images with captions removed",
                        type=str, default="./clean_covers")
    parser.add_argument("--checkpoint_root", help="Checkpoint location", type=str, default="./checkpoint")
    parser.add_argument("--lr", help="Learning rate", type=float, default=0.001)
    parser.add_argument("--epochs", help="Number of epochs to train for", type=int, default=138)
    parser.add_argument("--batch_size", help="Batch size", type=int, default=64)
    parser.add_argument("--canvas_size", help="Image canvas size for learning", type=int, default=256)
    parser.add_argument("--display_steps", help="How often to plot the samples", type=int, default=10)
    parser.add_argument("--plot_grad", help="Whether to plot the gradients", default=False, action="store_true")
    args = parser.parse_args()
    print(args)

    # Network properties
    num_conv_layers = 3
    num_linear_layers = 2

    # Plot properties
    bin_steps = 20  # How many steps to aggregate with mean for each plot point

    logger.info("--- Starting captioner_main ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataloader = get_train_data(
        args.checkpoint_root, args.original_covers, args.clean_covers,
        args.batch_size, args.canvas_size
    )

    logger.info("--- Captioner training ---")
    captioner = make_models(
        canvas_size=args.canvas_size,
        num_conv_layers=num_conv_layers,
        num_linear_layers=num_linear_layers,
        device=device
    )
    train(dataloader, captioner, device, {
        # Common
        "display_steps": args.display_steps,
        "bin_steps": bin_steps,
        "checkpoint_root": args.checkpoint_root,
        "n_epochs": args.epochs,
        "lr": args.lr,
        "plot_grad": args.plot_grad,
    })


if __name__ == '__main__':
    main()
