import json
import logging
import os
import pickle
from enum import Enum
from itertools import repeat
from pathlib import Path
from typing import Dict, Tuple

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm.contrib.concurrent import process_map

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms.functional import to_pil_image, to_tensor, rgb_to_grayscale

from kornia.color import rgb_to_rgba, rgb_to_grayscale
from kornia.morphology import dilation, erosion

from utils.bboxes import (
    BBox,
    find_bboxes,
    remove_overlaps,
    filter_bboxes_by_size,
    merge_aligned_bboxes,
    crop_bboxes_by_canvas,
    sort_bboxes_by_area
)
from utils.color_extractor import extract_primary_color
from utils.edges import detect_edges

logger = logging.getLogger("dataset")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.WARNING)

IMAGE_EXTENSION = 'jpg'

color_type = Tuple[int, int, int]
pos_type = Tuple[float, float, float, float]


def filename_extension(file_path: str) -> str:
    ext_with_dot = os.path.splitext(file_path)[1]
    return ext_with_dot[1:]


def replace_extension(file_path: str, new_extension: str) -> str:
    last_dot = file_path.rfind('.')
    return file_path[:last_dot + 1] + new_extension


class AugmentTransform(Enum):
    NONE = 0
    FLIP_H = 1
    ROTATE_90_CW = 2
    ROTATE_90_CCW = 3


def image_file_to_tensor(file_path: str, canvas_size: int,
                         transform: AugmentTransform = AugmentTransform.NONE) -> torch.Tensor:
    logger.info(f'Reading image file {file_path} to a tensor')
    # Convert to RGB as input can sometimes be grayscale
    image = Image.open(file_path).resize((canvas_size, canvas_size)).convert('RGB')

    if transform == AugmentTransform.FLIP_H:
        image = image.transpose(Image.FLIP_LEFT_RIGHT)
    elif transform == AugmentTransform.ROTATE_90_CW:
        image = image.rotate(270)
    elif transform == AugmentTransform.ROTATE_90_CCW:
        image = image.rotate(90)

    return to_tensor(image)  # CxHxW


def read_cover_tensor_for_file(f: str, cover_dir: str, canvas_size: int, transform: AugmentTransform):
    cover_file = f'{cover_dir}/{replace_extension(f, IMAGE_EXTENSION)}'
    assert os.path.isfile(cover_file), f'No cover image on disk for {f}, aborting!'
    cover_tensor = image_file_to_tensor(cover_file, canvas_size, transform)

    return cover_tensor


def rgb_to_plt(c: (int, int, int)) -> (float, float, float):
    r, g, b = c
    return r / 255, g / 255, b / 255


def plot_img_with_bboxes(img, bboxes_with_colors, mask=None, title=None):
    if mask is not None:
        pil_mask = to_pil_image(mask)
        plt.imshow(pil_mask)
        img = rgb_to_rgba(img, 0.5)

    pil_img = to_pil_image(img)
    plt.imshow(pil_img)
    if title is not None:
        plt.title(title)

    plt.xticks([])
    plt.yticks([])

    ax = plt.gca()
    for i, (b, c) in enumerate(bboxes_with_colors):
        plt.text(b.x1 - 10, b.y1 - 10, str(i + 1), backgroundcolor='w', fontsize=10.0)
        rect = Rectangle(
            (b.x1, b.y1), b.width(), b.height(),
            linewidth=1,
            edgecolor=rgb_to_plt(c), facecolor='none'
        )
        ax.add_patch(rect)

    plt.plot()
    plt.show()


def detect_label_bboxes_and_colors(f: str, original_cover_dir: str, clean_cover_dir: str,
                                   canvas_size: int) -> [[(BBox, color_type)]]:
    original_cover = image_file_to_tensor(f"{original_cover_dir}/{f}", canvas_size)
    clean_cover = image_file_to_tensor(f"{clean_cover_dir}/{f}", canvas_size)

    diff = rgb_to_grayscale(torch.abs(original_cover - clean_cover)).squeeze(0)  # HxW
    eps = 0.1
    diff = torch.where(diff > eps, diff, torch.zeros_like(diff))  # A mask of significant differences
    edit_mask = diff
    k = 21
    kernel = torch.ones(k, k)
    diff = diff.unsqueeze(dim=0).unsqueeze(dim=0)
    diff = dilation(diff, kernel)
    diff = erosion(diff, kernel).squeeze()

    bboxes = remove_overlaps(find_bboxes(diff))
    bboxes = filter_bboxes_by_size(bboxes, 2)
    bboxes = merge_aligned_bboxes(bboxes, canvas_size)  # Distant aligned letters -> words
    crop_bboxes_by_canvas(bboxes, canvas_size)
    bboxes = filter_bboxes_by_size(bboxes, 5)
    bboxes = sort_bboxes_by_area(bboxes)

    edit_mask = edit_mask.unsqueeze(dim=2).repeat((1, 1, 4))  # grayscale -> RGBA
    original_cover_masked = rgb_to_rgba(original_cover, 1.0).permute(1, 2, 0)  # CHW -> HWC
    invisible_background = torch.zeros_like(original_cover_masked)
    original_cover_masked = torch.where(edit_mask > 0, original_cover_masked, invisible_background)

    bboxes_filtered, colors = [], []
    for b in bboxes:
        original_fragment = original_cover_masked[b.y1:b.y2 + 1, b.x1:b.x2 + 1]
        primary_color = extract_primary_color(original_fragment, count=3)
        if primary_color is not None:
            bboxes_filtered.append(b)
            colors.append(primary_color)
    bboxes_with_colors = list(zip(bboxes_filtered, colors))

    return bboxes_with_colors


def markup_captions(dataset_files: [str],
                    extracted_bboxes_with_colors: [[(BBox, color_type)]],
                    original_cover_dir: str,
                    markup_cache_filename: str,
                    canvas_size: int) -> Dict[str, Tuple[pos_type, color_type, pos_type, color_type]]:
    data = {}

    # Restore incomplete markup from cache, if any
    if os.path.isfile(markup_cache_filename):
        with open(markup_cache_filename, 'r') as markup_cache_file:
            data = json.load(markup_cache_file)

    print_help_message = True

    for f, bboxes_with_colors in zip(dataset_files, extracted_bboxes_with_colors):
        if f in data:
            continue
        if print_help_message:
            print("*** Type `-1` if automatic detection is incorrect and captioning will be done manually\n"
                  "*** Type `0` if this cover should be excluded from dataset\n"
                  "*** Type `1` if the only box consists of artist and track name together\n"
                  "*** Or type the numbers of artist box and track name box (separated by comma) to capture")
            print_help_message = False

        original_cover = image_file_to_tensor(f"{original_cover_dir}/{f}", canvas_size)
        plot_img_with_bboxes(original_cover, bboxes_with_colors, mask=None, title=f)

        choice = None
        while choice is None:
            s = input("> ")
            if not s:
                continue
            s = [x.strip() for x in s.split(',')]
            if not all(s):
                continue
            try:
                s = list(map(int, s))
            except ValueError:
                continue
            if (s == [-1]) or (s == [0]) or (1 <= len(s) <= 2 and all([1 <= x <= len(bboxes_with_colors) for x in s])):
                choice = s

        if choice == [-1]:
            # Incorrect automatic detection, to be done manually
            result = []
        elif choice == [0]:
            # No labels, should be excluded from the dataset
            result = 0
        elif len(choice) == 1:
            # Artist + track name together
            choice = choice[0] - 1
            b, color = bboxes_with_colors[choice]
            pos = b.to_pos(canvas_size)
            result = pos, color, pos, color
        else:
            # Artist and track name separately
            choice = [c - 1 for c in choice]
            b_color = [bboxes_with_colors[c] for c in choice]
            art_b, art_c, track_b, track_c = b_color[0] + b_color[1]
            result = art_b.to_pos(canvas_size), art_c, track_b.to_pos(canvas_size), track_c

        data[f] = result
        # Aggressively cache markup results to prevent work loss
        with open(markup_cache_filename, 'w') as markup_cache_file:
            json.dump(data, markup_cache_file, indent=4, sort_keys=True, ensure_ascii=False)

    return data


def transform_bbox(cover_transform: AugmentTransform, bbox: [float]):
    x, y, w, h = bbox
    if cover_transform == AugmentTransform.NONE:
        return [x, y, w, h]
    elif cover_transform == AugmentTransform.FLIP_H:
        return [1 - x - w, y, w, h]
    elif cover_transform == AugmentTransform.ROTATE_90_CW:
        return [1 - y - h, x, h, w]
    elif cover_transform == AugmentTransform.ROTATE_90_CCW:
        return [y, 1 - x - w, h, w]


class CaptionDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, checkpoint_dir: str, original_cover_dir: str, clean_cover_dir: str,
                 canvas_size: int, should_cache: bool = True, augment: bool = True):
        self.checkpoint_root_ = f'{checkpoint_dir}/caption_dataset'
        self.cache_ = {} if should_cache else None
        self.augment_ = augment

        data_filename = f'{self.checkpoint_root_}/data.json'
        if os.path.isfile(data_filename):
            with open(data_filename, 'r') as f:
                data = json.load(f)
            # Omit entries marked as skipped
            entries_to_skip = [k for k in data.keys() if data[k] == 0]
            for k in entries_to_skip:
                data.pop(k)
            dataset_files = sorted(data.keys())
            for f in dataset_files:
                entry = data[f]
                assert len(entry) == 4, f"Malformed entry of wrong size: {entry}"
                assert len(entry[0]) == 4 and len(entry[2]) == 4, f"Positions are not 4 numbers for {f}"
                assert len(entry[1]) == 3 and len(entry[3]) == 3, f"Colors are not 3 numbers for {f}"
            for f in dataset_files:
                cover_file = f'{original_cover_dir}/{f}'
                assert os.path.isfile(cover_file), f'No original cover for {f}'
            logger.info(f'Dataset considered complete with {len(dataset_files)} covers.')
        else:
            logger.info('Building the dataset from original and clean covers')
            dataset_files = sorted([
                f for f in os.listdir(clean_cover_dir)
                if os.path.isfile(f'{clean_cover_dir}/{f}')
                   and filename_extension(f) == IMAGE_EXTENSION
            ])
            for f in dataset_files:
                original_cover_file = f'{original_cover_dir}/{f}'
                assert os.path.isfile(original_cover_file), f'No original cover for {f}'

            Path(self.checkpoint_root_).mkdir(exist_ok=True)
            detection_cache_filename = f'{self.checkpoint_root_}/detected.pickle'
            if os.path.isfile(detection_cache_filename):
                with open(detection_cache_filename, 'rb') as f:
                    extracted_bboxes_with_colors = pickle.load(f)
            else:
                extracted_bboxes_with_colors = process_map(
                    detect_label_bboxes_and_colors,
                    dataset_files,
                    repeat(original_cover_dir),
                    repeat(clean_cover_dir),
                    repeat(canvas_size),
                    max_workers=8,
                    chunksize=10
                )
                with open(detection_cache_filename, 'wb') as f:
                    pickle.dump(extracted_bboxes_with_colors, f)

            markup_cache_filename = f'{self.checkpoint_root_}/cache.json'
            data = markup_captions(
                dataset_files,
                extracted_bboxes_with_colors,
                original_cover_dir,
                markup_cache_filename,
                canvas_size
            )
            logger.info('Writing the dataset data.')
            with open(data_filename, 'w') as f:
                json.dump(data, f, indent=4, sort_keys=True, ensure_ascii=False)

        self.data_ = data
        self.dataset_files_ = dataset_files

        self.original_cover_dir_ = original_cover_dir
        self.canvas_size_ = canvas_size

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.cache_ is not None and index in self.cache_:
            return self.cache_[index]

        if self.augment_:
            cover_index = index // len(AugmentTransform)
            cover_transform = AugmentTransform(index % len(AugmentTransform))
        else:
            cover_index = index
            cover_transform = AugmentTransform.NONE
        f = self.dataset_files_[cover_index]

        cover_tensor = read_cover_tensor_for_file(f, self.original_cover_dir_, self.canvas_size_, cover_transform)
        edges = detect_edges(cover_tensor)
        data_entry = self.data_[f]
        artist_name_pos = torch.tensor(transform_bbox(cover_transform, data_entry[0]))
        artist_name_color = torch.tensor(data_entry[1]) / 255
        track_name_pos = torch.tensor(transform_bbox(cover_transform, data_entry[2]))
        track_name_color = torch.tensor(data_entry[3]) / 255

        pos_tensor = torch.cat((artist_name_pos, track_name_pos))
        color_tensor = torch.cat((artist_name_color, track_name_color))
        result = cover_tensor, edges, pos_tensor, color_tensor

        if self.cache_ is not None:
            self.cache_[index] = result

        return result

    def __len__(self) -> int:
        result = len(self.dataset_files_)
        if self.augment_:
            result *= len(AugmentTransform)
        return result
