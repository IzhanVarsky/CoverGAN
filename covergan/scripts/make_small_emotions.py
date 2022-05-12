import json
import os.path

with open('../emotions.json', encoding="utf-8") as f:
    track_songs = json.load(f)
sortedDict = list(filter(lambda fname: os.path.isfile(f'../small_audio/{fname[0]}'),
                         track_songs))

with open("../small_emotions.json", "w", encoding="utf-8") as f:
    json.dump(sortedDict, f)
