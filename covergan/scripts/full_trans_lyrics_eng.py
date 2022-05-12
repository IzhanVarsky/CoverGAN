import os
import re
from googletrans import Translator

translator = Translator()

files = os.listdir("./all_lyrics")
os.makedirs("./translated_all_lyrics_xxx", exist_ok=True)


def fltr(s):
    x1 = ''.join(filter(lambda c: not c.isdigit(), s))
    return re.sub('\\s+', ' ', x1).strip()


for f_name in files:
    with open(f"./all_lyrics/{f_name}", 'r', encoding='utf-8') as f:
        all_lines = f.readlines()
        all_lines[0] = all_lines[0].split("Lyrics")[1]
        all_lines[-1] = all_lines[-1].replace("Embed", "")
        all_lines = list(map(fltr, all_lines))
    f_name = f_name.replace("{!!!}", "")

    print(f_name)
    text = '\n'.join(all_lines)
    translation = translator.translate(text).text.split('\n')

    filtered = list(filter(lambda x: x.strip(), translation))
    filtered = set(filtered)
    if len(filtered) < 5:
        continue
    with open(f"./translated_all_lyrics_xxx/{f_name}", 'w', encoding='utf-8') as f:
        f.write('\n'.join(filtered))
