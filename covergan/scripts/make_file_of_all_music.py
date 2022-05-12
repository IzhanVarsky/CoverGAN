import json

with open('../dataset_full_covers/emotions.json', encoding="utf-8") as f:
    track_songs = json.load(f)
    res_str = ""
    for s in track_songs:
        res_str += s[0].replace(".mp3", "") + "\n"
    with open('full_songs.txt', encoding="utf-8", mode='w') as rf:
        rf.write(res_str)
    print(res_str)