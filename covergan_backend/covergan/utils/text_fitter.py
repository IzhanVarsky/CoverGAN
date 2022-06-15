from PIL import ImageFont, features

from .bboxes import BBox


# assert features.check('raqm')


def binary_search(predicate, lower_bound, upper_bound, default_value=None):
    ret = default_value
    while lower_bound + 1 < upper_bound:
        mid = lower_bound + (upper_bound - lower_bound) // 2
        if predicate(mid):
            ret = lower_bound = mid
        else:
            upper_bound = mid
    return ret


def get_text_wh(font: ImageFont.FreeTypeFont, text: str, direction: str):
    left, top, right, bottom = font.getbbox(text, direction, anchor='lt')
    width = right - left
    height = bottom - top
    if direction == 'ttb':
        width, height = height, width
    return width, height


def fit_text(text: str, boundary: BBox, font_filename: str) -> (int, str, int, int, float):
    boundary_width, boundary_height = boundary.width(), boundary.height()
    pos_wh_ratio = boundary.wh_ratio()
    direction = 'ttb' if pos_wh_ratio < 1 else 'ltr'

    def fits(fs: int):
        f = ImageFont.truetype(font_filename, int(fs))
        text_w, text_h = get_text_wh(f, text, direction)
        return text_w <= boundary_width and text_h <= boundary_height

    cannot_fit = False
    font_size = binary_search(fits, 10, 80)
    if font_size is None:
        cannot_fit = True
        # The predicted area is too small, so the text won't fit anyway; make it readable.
        font_size = 20

    font = ImageFont.truetype(font_filename, font_size)
    text_width, text_height = get_text_wh(font, text, direction)

    x_shift = (boundary_width - text_width) // 2
    y_shift = (boundary_height - text_height) // 2
    x = boundary.x1 + x_shift
    y = boundary.y1 + y_shift
    if direction == 'ttb':
        x += text_width // 2
    else:
        y += text_height

    if cannot_fit:
        score = 0
    else:
        score = text_width * text_height / boundary.area()

    return font_size, direction, x, y, score
