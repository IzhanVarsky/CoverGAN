import json
import os

import lyricsgenius as lg  # https://github.com/johnwmillr/LyricsGenius

genius = lg.Genius('QPDtjHNocsuNZ6vEDEclGdoL-KMDs04td0jDBrwynstLMKOV9kDzszBksJVWGbSg',
                   # Client access token from Genius Client API page
                   skip_non_songs=True, excluded_terms=["(Remix)", "(Live)"],
                   remove_section_headers=True)

os.makedirs("./all_lyrics", exist_ok=True)
already_made = os.listdir("./all_lyrics")
already_made.sort()

with open('../dataset_full_covers/emotions_orig.json', encoding="utf-8") as f:
    track_songs = json.load(f)
    full_songs = []
    for s in track_songs:
        name = s[0].replace(".mp3", "")
        # if len(already_made) != 0 and name <= already_made[-1]:
        #     continue
        full_songs.append(name)

for s in full_songs:
    splt = s.split(" - ")
    s_artist = splt[0]
    s_title = ' - '.join(splt[1:])

    song = genius.search_song(s_title, s_artist)
    if song is None:
        print("----- No song found :(")
        continue
    if song.artist.lower().replace(" ", "") not in s_artist.lower().replace(" ", "") and \
            s_artist.lower().replace(" ", "") not in song.artist.lower().replace(" ", ""):
        print(f"----- Bad artist: `{s_artist}` / `{song.artist}`.")
        continue
    warn = ""
    if song.title.lower().replace(" ", "").replace("'", "").replace("’", "") not in \
            s_title.lower().replace(" ", "").replace("'", "").replace("’", "") and \
            s_title.lower().replace(" ", "").replace("'", "").replace("’", "") not in \
            song.title.lower().replace(" ", "").replace("'", "").replace("’", ""):
        print(f"!!!!! Warning: `{s_title}` / `{song.title}`")
        warn = "{!!!}"
    with open(f'./all_lyrics/{s.replace(":", "_").replace("?", "_")}{warn}.txt', encoding="utf-8", mode='w') as f:
        f.write(song.lyrics)
    print("+++++++")
