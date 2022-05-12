import os

from googletrans import Translator

# init the Google API translator
translator = Translator()

# translate a spanish text to english text (by default)

os.makedirs("./translated_all_lyrics", exist_ok=True)

for f_name in os.listdir("./reformatted_all_lyrics"):
    print(f_name)
    with open("./reformatted_all_lyrics/" + f_name, "r") as f:
        text = f.read()
    translation = translator.translate(text)
    with open(f"./translated_all_lyrics/{f_name}", 'w') as f:
        f.write(translation.text)
    # print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
