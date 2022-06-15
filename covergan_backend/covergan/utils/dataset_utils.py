import os
import logging
from itertools import repeat
import torch
from multiprocessing import Pool
from pathlib import Path
from outer.metadata_extractor import get_tag_map

logger = logging.getLogger("dataset_utils")
logger.addHandler(logging.StreamHandler())
# logger.setLevel(logging.INFO)
logger.setLevel(logging.WARNING)

MUSIC_EXTENSIONS = ['flac', 'mp3']
IMAGE_EXTENSION = '.jpg'


def filename_extension(file_path: str) -> str:
    ext_with_dot = os.path.splitext(file_path)[1]
    return ext_with_dot[1:]


def replace_extension(file_path: str, new_extension: str) -> str:
    last_dot = file_path.rfind('.')
    return file_path[:last_dot] + new_extension


def get_tensor_file(checkpoint_root: str, f: str):
    return f'{checkpoint_root}/{f}.pt'


def get_cover_from_tags(f):
    return get_tag_map(f)['cover']


def process_music_file_to_tensor(f: str, checkpoint_root: str, audio_dir: str, cover_dir: str, num: int = None):
    music_tensor_f = get_tensor_file(checkpoint_root, f)
    if not os.path.isfile(music_tensor_f):
        create_and_save_music_tensor(audio_dir, f, music_tensor_f, num)

    cover_file = f'{cover_dir}/{replace_extension(f, IMAGE_EXTENSION)}'
    if not os.path.isfile(cover_file):
        logger.info(f'No cover file for {f}, attempting extraction...')
        music_file = f'{audio_dir}/{f}'
        image = get_cover_from_tags(music_file)
        assert image is not None, f'Failed to extract cover for {f}, aborting!'
        image.save(cover_file)
        logger.info(f'Cover image for {f} extracted and saved')


def create_and_save_music_tensor(audio_dir, f, music_tensor_f, num=None):
    logger.info(f'No tensor for {f}, generating...')
    music_file = f'{audio_dir}/{f}'
    create_and_save_music_file_to_tensor_inner(music_file, music_tensor_f, num)


def create_audio_tensors_for_folder(folder_path, out_folder):
    files = os.listdir(folder_path)
    os.makedirs(out_folder, exist_ok=True)
    for f in files:
        create_and_save_music_file_to_tensor_inner(f"{folder_path}/{f}", f"{out_folder}/{f}.pt")


def create_and_save_music_file_to_tensor_inner(input_file, out_file, num=None):
    from outer.audio_extractor import audio_to_embedding
    embedding = torch.from_numpy(audio_to_embedding(input_file, num))
    torch.save(embedding, out_file)


def create_music_tensor_files(checkpoint_root_, audio_dir, cover_dir):
    completion_marker = f'{checkpoint_root_}/COMPLETE'
    if os.path.isfile(completion_marker):
        dataset_files = sorted([
            f[:-len('.pt')] for f in os.listdir(checkpoint_root_)
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
            if os.path.isfile(f'{audio_dir}/{f}') and filename_extension(f) in MUSIC_EXTENSIONS
        ])

        Path(checkpoint_root_).mkdir(exist_ok=True)
        with Pool(maxtasksperchild=50) as pool:
            pool.starmap(
                process_music_file_to_tensor,
                zip(dataset_files, repeat(checkpoint_root_), repeat(audio_dir), repeat(cover_dir),
                    [i for i in range(len(dataset_files))]),
                chunksize=100
            )
        logger.info('Marking the dataset complete.')
        Path(completion_marker).touch(exist_ok=False)
    return dataset_files
