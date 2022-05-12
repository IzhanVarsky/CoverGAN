import json
import os
import shutil

base_old_dir = '../dataset_emoji_4'
base_new_dir = '../dataset_emoji_52'
os.makedirs(base_new_dir, exist_ok=True)
os.makedirs(f"{base_new_dir}/audio", exist_ok=True)
os.makedirs(f"{base_new_dir}/clean_covers", exist_ok=True)

audios = os.listdir(f"{base_old_dir}/audio")
covers = os.listdir(f"{base_old_dir}/clean_covers")

cnt_per_one_track = 13

for f_name in audios:
    for i in range(cnt_per_one_track):
        new_f_name = f_name.replace(".mp3", f"_{i}")
        shutil.copy(f"{base_old_dir}/audio/{f_name}",
                    f"{base_new_dir}/audio/{new_f_name}.mp3")
        shutil.copy(f"{base_old_dir}/clean_covers/{f_name.replace('.mp3', '.jpg')}",
                    f"{base_new_dir}/clean_covers/{new_f_name}.jpg")

with open(f'{base_old_dir}/emotions.json', encoding="utf-8") as f:
    track_songs = json.load(f)
    res = []
    for x in track_songs:
        for i in range(cnt_per_one_track):
            new_f_name = x[0].replace(".mp3", f"_{i}.mp3")
            res.append([new_f_name, x[1]])
    with open(f'{base_new_dir}/emotions.json', mode='w') as res_f:
        json.dump(res, res_f)
