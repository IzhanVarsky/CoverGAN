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
