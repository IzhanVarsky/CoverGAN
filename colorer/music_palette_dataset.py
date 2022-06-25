from typing import Optional, Tuple

import numpy as np
from torch.utils.data.dataset import Dataset

from outer.emotions import emotions_one_hot, read_emotion_file
from utils.dataset_utils import *
from utils.filenames import normalize_filename
from utils.image_clustering import cluster

logger = logging.getLogger("dataset")
logger.addHandler(logging.StreamHandler())
# logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)


def get_main_rgb_palette(f_path: str, color_count: int, quality=5):
    from colorthief import ColorThief
    color_thief = ColorThief(f_path)
    return color_thief.get_palette(color_count=color_count + 1, quality=quality)


def get_main_rgb_palette2(f_path: str, color_count: int, sort_colors=True):
    import PIL.Image
    pil_image = PIL.Image.open(f_path).convert(mode='RGB')
    labels, centers = cluster(np.asarray(pil_image), k=color_count, only_labels_centers=True)
    if not sort_colors:
        return [list(x) for x in centers]

    from collections import Counter
    dict = Counter(labels.flatten())
    sorted_labels = sorted(dict.items(), key=lambda x: x[1], reverse=True)
    palette = [list(centers[label[0]]) for label in sorted_labels]
    return palette


def image_file_to_palette(f: str, cover_dir: str, color_count: int, color_type: str = 'rgb',
                          sorted_colors: bool = True):
    cover_file = f'{cover_dir}/{replace_extension(f, IMAGE_EXTENSION)}'
    if sorted_colors:
        palette = get_main_rgb_palette2(cover_file, color_count, sort_colors=True)
    else:
        palette = get_main_rgb_palette(cover_file, color_count)
    palette = [list(t) for t in palette]
    palette = palette + palette[:color_count - len(palette)]
    assert len(palette) == color_count

    if color_type == 'rgb':
        return palette
    if color_type == 'lab':
        from colorer.colors_transforms import rgb_to_cielab, cielab_rgb_to
        print("before:", palette)
        palette_very_before = palette.copy()
        palette = [rgb_to_cielab(np.array(p)) for p in palette]
        print("after:", palette)
        palette_old_check = [cielab_rgb_to(np.array(p)) for p in palette]
        palette_old_check2 = 255 * np.array(palette_old_check)
        print("after:", palette_old_check)
        print("after:", palette_old_check2)
        return palette


def process_cover_to_palette(f: str, checkpoint_root: str, cover_dir: str, palette_count: int,
                             color_type: str = 'rgb', sorted_colors: bool = True, num: int = None):
    palette_tensor_f = get_tensor_file(checkpoint_root, replace_extension(f, ""))
    if not os.path.isfile(palette_tensor_f):
        logger.info(f'No palette tensor for file #{num}: {f}, generating...')
        palette = image_file_to_palette(f, cover_dir, palette_count,
                                        color_type=color_type, sorted_colors=sorted_colors)
        palette = np.concatenate(palette)  # / 255
        palette_tensor = torch.from_numpy(palette).float()
        torch.save(palette_tensor, palette_tensor_f)


def read_tensor_from_file(f: str, checkpoint_root: str):
    tensor_f = get_tensor_file(checkpoint_root, f)
    assert os.path.isfile(tensor_f), f'Tensor file missing for {tensor_f}'
    return torch.load(tensor_f)


class MusicPaletteDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, name: str, checkpoint_dir: str, audio_dir: str, cover_dir: str, emotion_file: Optional[str],
                 sort_colors: bool = True,
                 should_cache: bool = True,
                 is_for_train: bool = True, train_test_split_coef: float = 0.9):
        self.color_type = 'rgb'
        # self.color_type = 'lab'
        self.sorted_color = 'sorted' if sort_colors else 'unsorted'
        self.palette_count = 12
        self.palette_name = 'palette_dataset'
        self.checkpoint_root_ = f'{checkpoint_dir}/{name}'
        self.palette_checkpoint_root_ = f'{checkpoint_dir}/{self.palette_name}/' \
                                        f'palette_{self.color_type}_count_{self.palette_count}_{self.sorted_color}'
        self.cache_ = {} if should_cache else None

        self.is_for_train = is_for_train
        self.train_test_split_coef = train_test_split_coef

        self.create_palette_tensor_files(cover_dir)
        create_music_tensor_files(self.checkpoint_root_, audio_dir, cover_dir)

        self.dataset_files_ = self.get_dataset_files()

        self.emotions_dict_ = None
        if emotion_file is not None:
            if not os.path.isfile(emotion_file):
                print(f"WARNING: Emotion file '{emotion_file}' does not exist")
            else:
                emotions_list = read_emotion_file(emotion_file)
                emotions_dict = dict(emotions_list)
                self.emotions_dict_ = emotions_dict
                for f in self.dataset_files_:
                    f = normalize_filename(f)
                    if f not in emotions_dict:
                        print(f"Emotions were not provided for dataset file {f}")
                        self.emotions_dict_ = None
                if self.emotions_dict_ is None:
                    print("WARNING: Ignoring emotion data, see reasons above.")
                else:
                    for f, emotions in self.emotions_dict_.items():
                        self.emotions_dict_[f] = emotions_one_hot(emotions)

        self.cover_dir_ = cover_dir

    def get_dataset_files(self):
        dataset_files = sorted([
            f[:-len('.pt')] for f in os.listdir(self.checkpoint_root_)
            if f.endswith('.pt')
        ])
        for_train_files_count = int(len(dataset_files) * self.train_test_split_coef)
        print(f"--- {for_train_files_count} files considered for training")
        print(f"--- {len(dataset_files) - for_train_files_count} files considered for testing")
        if self.is_for_train:
            dataset_files = dataset_files[:for_train_files_count]
        else:
            dataset_files = dataset_files[for_train_files_count:]
        return dataset_files

    def create_palette_tensor_files(self, cover_dir: str):
        completion_marker = f'{self.palette_checkpoint_root_}/COMPLETE'
        if not os.path.isfile(completion_marker):
            logger.info('Building the palette dataset based on covers')
            dataset_files = sorted([
                f for f in os.listdir(cover_dir)
                if os.path.isfile(f'{cover_dir}/{f}') and filename_extension(f) in IMAGE_EXTENSION
            ])
            os.makedirs(self.palette_checkpoint_root_, exist_ok=True)
            with Pool(maxtasksperchild=50) as pool:
                pool.starmap(
                    process_cover_to_palette,
                    zip(dataset_files, repeat(self.palette_checkpoint_root_), repeat(cover_dir),
                        repeat(self.palette_count),
                        repeat(self.color_type),
                        repeat(self.sorted_color),
                        [i for i in range(len(dataset_files))]),
                    chunksize=100
                )
            logger.info('Marking the palette dataset complete.')
            Path(completion_marker).touch(exist_ok=False)
        else:
            dataset_files = sorted([
                f[:-len('.pt')] for f in os.listdir(self.palette_checkpoint_root_)
                if f.endswith('.pt')
            ])
            for f in dataset_files:
                cover_file = f'{cover_dir}/{f}{IMAGE_EXTENSION}'
                assert os.path.isfile(cover_file), f'No cover for {f}'
            logger.info(f'Palette dataset considered complete with {len(dataset_files)} covers.')

    def has_emotions(self):
        return self.emotions_dict_ is not None

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cache_ is not None and index in self.cache_:
            return self.cache_[index]

        track_index = index
        f = self.dataset_files_[track_index]

        music_tensor = read_tensor_from_file(f, self.checkpoint_root_)
        palette_tensor = read_tensor_from_file(replace_extension(f, ""), self.palette_checkpoint_root_)
        emotions = self.emotions_dict_[normalize_filename(f)] if self.emotions_dict_ is not None else None

        target_count = 24  # 2m = 120s, 120/5
        if len(music_tensor) < target_count:
            music_tensor = music_tensor.repeat(target_count // len(music_tensor) + 1, 1)
        music_tensor = music_tensor[:target_count]

        if emotions is not None:
            result = music_tensor, palette_tensor, emotions
            # result = music_tensor, palette_tensor, emotions, f
        else:
            result = music_tensor, palette_tensor

        if self.cache_ is not None:
            self.cache_[index] = result

        return result

    def __len__(self) -> int:
        return len(self.dataset_files_)
