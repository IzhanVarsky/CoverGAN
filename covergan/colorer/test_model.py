#!/usr/bin/env python3
# coding: utf-8
import random

import numpy as np
import torch
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt, patches
from torch.utils.data import DataLoader

from colorer.colors_transforms import rgb_lab_rgb, rgb_to_cielab, cielab_rgb_to
from colorer.models.colorer import Colorer
from colorer.music_palette_dataset import MusicPaletteDataset, image_file_to_palette
from colorer_train import file_in_folder
from outer.emotions import emotions_one_hot, Emotion
from utils.noise import get_noise


def get_train_data(checkpoint_dir: str, audio_dir: str, cover_dir: str, emotion_file: str,
                   batch_size: int, augment_dataset: bool) -> (DataLoader, int, (int, int, int), bool):
    dataset = MusicPaletteDataset("cgan_out_dataset", checkpoint_dir,
                                  audio_dir, cover_dir, emotion_file,
                                  augment_dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    music_tensor, palette_tensor = dataset[0][:2]
    audio_embedding_dim = music_tensor.shape[1]
    palette_shape = palette_tensor.shape
    has_emotions = dataset.has_emotions()

    return dataloader, audio_embedding_dim, palette_shape, has_emotions


def main():
    f_names = [
        "&me - The Rapture Pt.II.mp3",
        "Zomboy - Lone Wolf.mp3",
        "Zero 7 - Home.mp3",
        "ZAYSTIN - Without You.mp3",
        "Yu Jae Seok - Dancing King.mp3",
        "Vargas & Lagola - Selfish.mp3",
        "voisart - Like Glass.mp3",
        "Кино - Группа крови.mp3",
        "Кино - Звезда по имени Солнце.mp3",
    ]
    for f_name in f_names:
        compare_for_file(f_name, 12)


def compare_for_file(f_name, colors_count):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    disc_slices = 6
    num_samples = 1
    emotions = [random.choice(list(Emotion))]
    deterministic = False
    z_dim = 32
    cover_file_name = f"../dataset_full_covers/clean_covers/{f_name.replace('.mp3', '.jpg')}"
    audio_checkpoint_fname = f"../dataset_full_covers/checkpoint/cgan_out_dataset/{f_name}.pt"
    # music_tensor = torch.from_numpy(audio_to_embedding(audio_file_name))
    music_tensor = torch.load(audio_checkpoint_fname)
    target_count = 24  # 2m = 120s, 120/5
    if len(music_tensor) < target_count:
        music_tensor = music_tensor.repeat(target_count // len(music_tensor) + 1, 1)
    music_tensor = music_tensor[:target_count].float().to(device)
    music_tensor = music_tensor[:disc_slices].flatten()
    music_tensor = music_tensor.unsqueeze(dim=0).repeat((num_samples, 1))
    emotions_tensor = emotions_one_hot(emotions).to(device).unsqueeze(dim=0).repeat((num_samples, 1))
    if deterministic:
        noise = torch.zeros((num_samples, z_dim), device=device)
    else:
        noise = get_noise(num_samples, z_dim, device=device)
    generator = get_palette_predictor()
    palette = generator.predict(noise, music_tensor, emotions_tensor)
    imsize = 512

    palette = [tuple(map(int, c)) for c in palette]
    im = Image.new('RGB', (imsize * 3, imsize))
    im.paste(Image.open(cover_file_name).resize((imsize, imsize)))
    draw = ImageDraw.Draw(im)

    real_palette = image_file_to_palette(f_name, "../dataset_full_covers/clean_covers", colors_count)
    print("=" * 20)
    print("real palette:             ", real_palette)
    rgb_lab_real_pal = [rgb_lab_rgb(p) for p in real_palette]
    print("rgb lab real palette:     ", rgb_lab_real_pal)
    print("lab real palette:         ", [list(rgb_to_cielab(p)) for p in real_palette])
    print("predicted:                ", [list(p) for p in palette])
    rgb_lab_pred_pal = [rgb_lab_rgb(p) for p in palette]
    print("rgb lab predicted palette:", rgb_lab_pred_pal)
    print("lab predicted palette:    ", [list(rgb_to_cielab(p)) for p in palette])

    # palettes_to_paint = [real_palette, rgb_lab_real_pal, palette, rgb_lab_pred_pal]
    palettes_to_paint = [real_palette, palette]
    for pal_i, pal in enumerate(palettes_to_paint):
        cur_h = 0
        width = imsize / (colors_count - 1)
        for i, p in enumerate(pal):
            p = tuple(p)
            colorval = "#%02x%02x%02x" % p
            draw.rectangle((imsize * (pal_i + 1), cur_h, imsize * (pal_i + 2) - 10, cur_h + width), fill=colorval)
            cur_h += width
    im.show()


def get_palette_predictor():
    import os
    # print("cur path:", os.path.abspath(os.getcwd()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    colors_count = 12
    disc_slices = 6
    z_dim = 32
    num_gen_layers = 5
    audio_embedding_dim = 281
    # gan_weights = f"../dataset_full_covers/checkpoint/colorer_{colors_count}_colors-28800.pt"
    # gan_weights = f"../dataset_full_covers/checkpoint/colorer_{colors_count}_colors-600.pt"
    if os.getcwd().endswith("covergan"):
        gan_weights = f"./dataset_full_covers/checkpoint/colorer_{colors_count}_colors-1800.pt"
    else:
        gan_weights = f"../dataset_full_covers/checkpoint/colorer_{colors_count}_colors-1800.pt"

    generator = Colorer(
        z_dim=z_dim,
        audio_embedding_dim=audio_embedding_dim * disc_slices,
        has_emotions=True,
        num_layers=num_gen_layers,
        colors_count=colors_count)
    generator.eval()
    gan_weights = torch.load(gan_weights, map_location=device)
    generator.load_state_dict(gan_weights["0_state_dict"])
    return generator


if __name__ == '__main__':
    main()
