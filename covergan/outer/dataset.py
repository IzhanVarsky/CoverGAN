import os
from enum import Enum
from itertools import repeat
from typing import Optional, Tuple
import logging
from multiprocessing import Pool
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from .metadata_extractor import get_tag_map
from .emotions import emotions_one_hot, read_emotion_file
from utils.filenames import normalize_filename

logger = logging.getLogger("dataset")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)

MUSIC_EXTENSIONS = ['flac', 'mp3']
IMAGE_EXTENSION = '.jpg'


class AugmentTransform(Enum):
    NONE = 0
    FLIP_H = 1
    FLIP_V = 2


def filename_extension(file_path: str) -> str:
    ext_with_dot = os.path.splitext(file_path)[1]
    return ext_with_dot[1:]


def replace_extension(file_path: str, new_extension: str) -> str:
    last_dot = file_path.rfind('.')
    return file_path[:last_dot] + new_extension


def image_file_to_tensor(file_path: str, canvas_size: int, transform: AugmentTransform) -> torch.Tensor:
    logger.info(f'Reading image file {file_path} to a tensor')
    # Convert to RGB as input can sometimes be grayscale
    image = Image.open(file_path).resize((canvas_size, canvas_size)).convert('RGB')
    if transform == AugmentTransform.FLIP_H:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform == AugmentTransform.FLIP_V:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return transforms.ToTensor()(image)  # CxHxW


def get_cover_from_tags(f):
    return get_tag_map(f)['cover']


def music_tensor_file(checkpoint_root: str, f: str):
    return f'{checkpoint_root}/{f}.pt'


def process_file(f: str, checkpoint_root: str, audio_dir: str, cover_dir: str, num: int = None):
    music_tensor_f = music_tensor_file(checkpoint_root, f)
    if not os.path.isfile(music_tensor_f):
        from .audio_extractor import audio_to_embedding

        logger.info(f'No tensor for {f}, generating...')
        music_file = f'{audio_dir}/{f}'
        embedding = torch.from_numpy(audio_to_embedding(music_file, num))
        torch.save(embedding, music_tensor_f)

    cover_file = f'{cover_dir}/{replace_extension(f, IMAGE_EXTENSION)}'
    if not os.path.isfile(cover_file):
        logger.info(f'No cover file for {f}, attempting extraction...')
        music_file = f'{audio_dir}/{f}'
        image = get_cover_from_tags(music_file)
        assert image is not None, f'Failed to extract cover for {f}, aborting!'
        image.save(cover_file)
        logger.info(f'Cover image for {f} extracted and saved')


def read_music_tensor_for_file(f: str, checkpoint_root: str):
    music_tensor_f = music_tensor_file(checkpoint_root, f)
    assert os.path.isfile(music_tensor_f), f'Music tensor file missing for {f}'
    music_tensor = torch.load(music_tensor_f)

    return music_tensor


def read_cover_tensor_for_file(f: str, cover_dir: str, canvas_size: int, image_transform: AugmentTransform):
    cover_file = f'{cover_dir}/{replace_extension(f, IMAGE_EXTENSION)}'
    assert os.path.isfile(cover_file), f'No cover image on disk for {f}, aborting!'
    cover_tensor = image_file_to_tensor(cover_file, canvas_size, image_transform)

    return cover_tensor


class MusicDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, name: str, checkpoint_dir: str, audio_dir: str, cover_dir: str, emotion_file: Optional[str],
                 canvas_size: int, augment: bool = False, should_cache: bool = True):
        self.checkpoint_root_ = f'{checkpoint_dir}/{name}'
        self.augment_ = augment
        self.cache_ = {} if should_cache else None

        completion_marker = f'{self.checkpoint_root_}/COMPLETE'
        if os.path.isfile(completion_marker):
            dataset_files = sorted([
                f[:-len('.pt')] for f in os.listdir(self.checkpoint_root_)
                if f.endswith('.pt')
            ])
            for f in dataset_files:
                cover_file = f'{cover_dir}/{replace_extension(f, IMAGE_EXTENSION)}'
                assert os.path.isfile(cover_file), f'No cover for {f}'
            logger.info(f'Dataset considered complete with {len(dataset_files)} tracks and covers.')
        else:
            logger.info('Building the dataset based on music')
            dataset_files = sorted([
                f for f in os.listdir(audio_dir)
                if os.path.isfile(f'{audio_dir}/{f}')
                   and filename_extension(f) in MUSIC_EXTENSIONS
            ])

            Path(self.checkpoint_root_).mkdir(exist_ok=True)
            with Pool(maxtasksperchild=50) as pool:
                pool.starmap(
                    process_file,
                    zip(dataset_files, repeat(self.checkpoint_root_), repeat(audio_dir), repeat(cover_dir),
                        [i for i in range(len(dataset_files))]),
                    chunksize=100
                )
            logger.info('Marking the dataset complete.')
            Path(completion_marker).touch(exist_ok=False)

        self.emotions_dict_ = None
        if emotion_file is not None:
            if not os.path.isfile(emotion_file):
                print(f"WARNING: Emotion file '{emotion_file}' does not exist")
            else:
                emotions_list = read_emotion_file(emotion_file)
                emotions_dict = dict(emotions_list)
                self.emotions_dict_ = emotions_dict
                for filename in dataset_files:
                    filename = normalize_filename(filename)
                    if filename not in emotions_dict:
                        print(f"Emotions were not provided for dataset file {filename}")
                        self.emotions_dict_ = None
                if self.emotions_dict_ is None:
                    print("WARNING: Ignoring emotion data, see reasons above.")
                else:
                    for filename, emotions in self.emotions_dict_.items():
                        self.emotions_dict_[filename] = emotions_one_hot(emotions)

        self.dataset_files_ = dataset_files
        self.cover_dir_ = cover_dir
        self.canvas_size_ = canvas_size

    def has_emotions(self):
        return self.emotions_dict_ is not None

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_ is not None and index in self.cache_:
            return self.cache_[index]

        if self.augment_:
            track_index = index // len(AugmentTransform)
            image_transform = AugmentTransform(index % len(AugmentTransform))
        else:
            track_index = index
            image_transform = AugmentTransform.NONE
        f = self.dataset_files_[track_index]

        music_tensor = read_music_tensor_for_file(f, self.checkpoint_root_)
        cover_tensor = read_cover_tensor_for_file(f, self.cover_dir_, self.canvas_size_, image_transform)
        emotions = self.emotions_dict_[normalize_filename(f)] if self.emotions_dict_ is not None else None

        target_count = 24  # 2m = 120s, 120/5
        if len(music_tensor) < target_count:
            music_tensor = music_tensor.repeat(target_count // len(music_tensor) + 1, 1)
        music_tensor = music_tensor[:target_count]

        if emotions is not None:
            result = music_tensor, cover_tensor, emotions
            # result = music_tensor, cover_tensor, emotions, f
        else:
            result = music_tensor, cover_tensor

        if self.cache_ is not None:
            self.cache_[index] = result

        return result

    def __len__(self) -> int:
        result = len(self.dataset_files_)
        if self.augment_:
            result *= len(AugmentTransform)
        return result
