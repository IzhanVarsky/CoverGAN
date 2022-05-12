import os
import shutil

from fontTools.ttLib import TTFont


def font_supports_all_glyphs(phrase_words, font_path):
    font = TTFont(font_path)
    for word in phrase_words:
        for c in word:
            if not has_glyph(font, c):
                return False
    return True


def has_glyph(font, glyph):
    for table in font['cmap'].tables:
        if ord(glyph) in table.cmap.keys():
            return True
    return False


fonts_dir = r"F:\stazhirovka2021\Diploma\covergan-test\covergan\fonts"

if __name__ == '__main__':
    non_unicode_fonts_dir = r"F:\stazhirovka2021\Diploma\covergan-test\covergan\fonts_non_unicode"
    os.makedirs(non_unicode_fonts_dir, exist_ok=True)
    fonts = os.listdir(fonts_dir)
    non_unicode_fonts = []
    for ind, f in enumerate(fonts):
        print(f"Processing font #{ind}: `{f}`...")
        try:
            font = TTFont(f'{fonts_dir}/{f}')
        except:
            print(f"!!!!!! Bad font: `{f}`")
            continue
        for c in "абвгдеёжзийклмнопрстуфхцчшщъыьэюяАБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ":
            if not has_glyph(font, c):
                print(f"------ Font `{f}` doesn't support symbol `{c}`.")
                non_unicode_fonts.append(f)
                break
    for x in non_unicode_fonts:
        shutil.move(f"{fonts_dir}/{x}", f"{non_unicode_fonts_dir}/{x}")
