#!/usr/bin/env python3
# coding: utf-8
import argparse
import base64
import csv
import os
import requests
from itertools import repeat
from multiprocessing import Pool
from typing import List, Optional, Tuple


EMOTIONS = [
    "anger",
    "comfortable",
    "fear",
    "funny",
    "happy",
    "inspirational",
    "joy",
    "lonely",
    "nostalgic",
    "passionate",
    "quiet",
    "relaxed",
    "romantic",
    "sadness",
    "serious",
    "soulful",
    "surprise",
    "sweet",
    "wary"
]


def convert_response(json_data: [(str, str)]):
    result = []
    for sample in json_data:
        svg_data = sample["svg"]
        png_str = sample["base64"]
        png_data = base64.b64decode(png_str)
        result.append((svg_data, png_data))
    return result


def send_api_request(address: str,
                     track_artist: str,
                     track_name: str,
                     emotion: str,
                     audio_file: str) -> Optional[List[Tuple[str, bytes]]]:
    emotion = emotion.lower()
    assert emotion in EMOTIONS
    try:
        url = f'{address}/generate'
        params = {
            'track_artist': track_artist,
            'track_name': track_name,
            'emotion': emotion
        }
        files = {
            'audio_file': open(audio_file, 'rb')
        }
        response = requests.post(url, params=params, files=files)
        try:
            return convert_response(response.json())
        except Exception as e:
            with open("failed-requests-log.txt", 'a+') as f:
                f.write(response.text)
            return None
    except requests.exceptions.RequestException as e:
        print(e)
        return None


def health(address: str):
    try:
        url = f'{address}/health'
        resp = requests.get(url).json()
        print(resp)
    except requests.exceptions.RequestException as e:
        print(e)


def ask_requests() -> [(str, str, str, str)]:
    result = []
    cmd = '?'
    while cmd:
        cmd = input('artist,track,emotion,file > ')
        cmd_split = cmd.split('|')
        if len(cmd_split) != 4:
            continue
        result.append(cmd_split)
    return result


def read_requests_file(requests_file: str) -> [(str, str, str, str)]:
    result = []
    with open(requests_file, newline='') as f:
        csv_reader = csv.reader(f, delimiter='|')
        for row in csv_reader:
            result.append(row)
    return result


def run_request(address, output_dir, req) -> bool:
    track_artist, track_name, emotion, audio_file = req
    if not os.path.isfile(audio_file):
        print(f'File not found: "{audio_file}", skipping.')
        return False

    print(f'{track_artist} - {track_name}, {emotion}, "{audio_file}"')
    health(address)
    resp = send_api_request(address, track_artist, track_name, emotion, audio_file)
    if resp is not None:
        resp_dir = f'{output_dir}/{track_artist} - {track_name}'
        os.makedirs(resp_dir, exist_ok=True)
        for j, (svg_data, png_data) in enumerate(resp):
            with open(f'{resp_dir}/{j}.svg', 'w') as f:
                f.write(svg_data)
            with open(f'{resp_dir}/{j}.png', 'wb') as f:
                f.write(png_data)
        return True

    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--address', help='CoverGAN server address', type=str, default='http://localhost:5001')
    parser.add_argument('--output_dir', help='Directory to save the results in', type=str, default='out')
    parser.add_argument('--requests', help='File with requests descriptions', type=str, default=None)
    parser.add_argument('--parallel', help='How many requests to run in parallel', type=int, default=2)
    args = parser.parse_args()

    address = args.address
    parallel_requests = args.parallel

    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    if args.requests is None:
        req = ask_requests()
    else:
        req = read_requests_file(args.requests)

    total_requests = len(req)
    with Pool(processes=parallel_requests) as pool:
        is_successful = pool.starmap(
            run_request,
            zip(repeat(address), repeat(output_dir), req),
            chunksize=100
        )
    health(address)
    successful_requests = sum(is_successful)
    print(f'Requests successful: {successful_requests}/{total_requests}')


if __name__ == '__main__':
    main()
