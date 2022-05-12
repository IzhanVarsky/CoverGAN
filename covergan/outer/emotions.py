from enum import IntEnum
import json
from typing import Optional

import torch


# IntEnum allows to compare enum values to ints directly
class Emotion(IntEnum):
    ANGER = 0
    COMFORTABLE = 1
    FEAR = 2
    FUNNY = 3
    HAPPY = 4
    INSPIRATIONAL = 5
    JOY = 6
    LONELY = 7
    NOSTALGIC = 8
    PASSIONATE = 9
    QUIET = 10
    RELAXED = 11
    ROMANTIC = 12
    SADNESS = 13
    SERIOUS = 14
    SOULFUL = 15
    SURPRISE = 16
    SWEET = 17
    WARY = 18

    def __str__(self) -> str:
        return self.name.lower()


def emotion_from_str(emotion_str: str) -> Optional[Emotion]:
    try:
        return Emotion[emotion_str.upper()]
    except KeyError:
        print(f"Unknown emotion: {emotion_str}")
        return None


def read_emotion_file(emotions_filename: str):
    with open(emotions_filename, 'r', encoding="utf-8") as f:
        emotion_list = json.load(f)
    for entry in emotion_list:
        if len(entry) != 2:
            print(f"Malformed entry in emotion file: {entry}")
            return None
    result = []
    for filename, emotion_strs in emotion_list:
        emotions = [emotion_from_str(x) for x in emotion_strs]
        if None in emotions:
            print(f"Unknown emotion in emotions list for dataset file '{filename}': {emotions}")
            return None
        if not (2 <= len(emotions) <= 3):
            print(f"Invalid emotion count for dataset file '{filename}'")
            return None
        result.append((filename, emotions))

    print(f"Successfully parsed emotion file with {len(result)} entries.")
    return result


def emotions_one_hot(emotions_list: [Emotion]) -> torch.Tensor:
    emotions_int_list = [int(x) for x in emotions_list]
    result = torch.zeros(len(Emotion))
    result[emotions_int_list] = 1
    return result
