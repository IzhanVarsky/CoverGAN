from unicodedata import normalize


# This is a workaround for Colab's Google Drive filename processing
def normalize_filename(filename: str):
    return normalize("NFC", filename)
