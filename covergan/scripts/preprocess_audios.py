#!/usr/bin/env python3
# coding: utf-8
import argparse
import numpy as np

from torch.utils.data.dataloader import DataLoader

from outer.dataset import MusicDataset
from outer.emotions import read_emotion_file


def get_dataset(checkpoint_dir: str, audio_dir: str, cover_dir: str, emotion_file: str,
                canvas_size: int,
                augment_dataset: bool):
    dataset = MusicDataset("cgan_out_dataset", checkpoint_dir, audio_dir, cover_dir, emotion_file, canvas_size,
                           augment_dataset)
    return DataLoader(dataset, batch_size=1, shuffle=False)


def file_in_folder(dir, file):
    if file is None:
        return None
    return f"{dir}/{file}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", help="Directory with all folders for training", type=str,
                        default="../dataset_small_20")
    parser.add_argument("--plots", help="Directory where save plots while training", type=str, default="plots")
    parser.add_argument("--audio", help="Directory with the music files", type=str, default="audio")
    parser.add_argument("--covers", help="Directory with the cover images", type=str, default="clean_covers")
    parser.add_argument("--emotions", help="File with emotion markup for train dataset", type=str,
                        default="emotions.json")
    parser.add_argument("--checkpoint_root", help="Checkpoint location", type=str, default="checkpoint")
    parser.add_argument("--augment_dataset", help="Whether to augment the dataset", default=False, action="store_true")
    parser.add_argument("--canvas_size", help="Image canvas size for learning", type=int, default=128)
    args = parser.parse_args()
    print(args)

    disc_slices = 6

    emotion_file = file_in_folder(args.train_dir, args.emotions)
    dataset = get_dataset(
        file_in_folder(args.train_dir, args.checkpoint_root),
        file_in_folder(args.train_dir, args.audio),
        file_in_folder(args.train_dir, args.covers),
        emotion_file,
        args.canvas_size, args.augment_dataset
    )
    res = []
    for batch in dataset:
        if len(batch) == 3:
            audio_embedding, real_cover_tensor, emotions = batch
        else:
            audio_embedding, real_cover_tensor = batch
            emotions = None
        cur_batch_size = len(audio_embedding)
        audio_embedding = audio_embedding.float()
        audio_embedding_disc = audio_embedding[:, :disc_slices].reshape(cur_batch_size, -1)

        if emotions is not None:
            inp = np.concatenate([audio_embedding_disc[0], emotions[0]])
        else:
            inp = audio_embedding_disc[0]
        res.append(inp)
    with open('res_embeds.txt', 'w') as f:
        for x in res:
            qwe = ""
            for y in x:
                qwe += str(y) + " "
            f.write(qwe + "\n")

    emotions_list = read_emotion_file(emotion_file)
    emotions_dict = dict(emotions_list)
    for x in emotions_dict:
        f = open('./embeds/' + x.replace('.mp3', '.txt'), 'w')
        res = ""
        for q in emotions_dict[x]:
            res += str(q) + " "
        f.write(res)


if __name__ == '__main__':
    main()
