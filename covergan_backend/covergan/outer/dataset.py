from enum import Enum
from itertools import repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Optional, Tuple

import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils.dataset_utils import *
from utils.filenames import normalize_filename
from .emotions import emotions_one_hot, read_emotion_file

logger = logging.getLogger("dataset")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)


class AugmentTransform(Enum):
    NONE = 0
    FLIP_H = 1
    FLIP_V = 2


def image_file_to_tensor(file_path: str, canvas_size: int, transform: AugmentTransform) -> torch.Tensor:
    logger.info(f'Reading image file {file_path} to a tensor')
    # Convert to RGB as input can sometimes be grayscale
    image = Image.open(file_path).resize((canvas_size, canvas_size)).convert('RGB')
    if transform == AugmentTransform.FLIP_H:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform == AugmentTransform.FLIP_V:
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
    return transforms.ToTensor()(image)  # CxHxW


def read_music_tensor_for_file(f: str, checkpoint_root: str):
    music_tensor_f = get_tensor_file(checkpoint_root, f)
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

        dataset_files = create_music_tensor_files(self.checkpoint_root_, audio_dir, cover_dir)

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
