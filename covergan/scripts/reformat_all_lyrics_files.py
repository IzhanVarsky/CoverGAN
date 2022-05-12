import os
import re

os.makedirs("./reformatted_all_lyrics", exist_ok=True)

files = os.listdir("./all_lyrics")

def fltr(s):
    x1 = ''.join(filter(lambda c: not c.isdigit(), s))
    return re.sub('\\s+', ' ', x1).strip()


for f_name in files:
    # if "{!!!}" in f_name:
    #     print(f_name)
    #     continue
    with open(f"./all_lyrics/{f_name}", 'r') as f:
        all_lines = f.readlines()
        all_lines[0] = all_lines[0].split("Lyrics")[1]
        all_lines[-1] = all_lines[-1].replace("Embed", "")
        all_lines = list(map(fltr, all_lines))
    f_name = f_name.replace("{!!!}", "")

    filtered = list(filter(lambda x: x.strip(), all_lines))
    filtered = set(filtered)
    if len(filtered) < 5:
        continue
    with open(f"./reformatted_all_lyrics/{f_name}", 'w') as f:
        f.write('\n'.join(filtered))
