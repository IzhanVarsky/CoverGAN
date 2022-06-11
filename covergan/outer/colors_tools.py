import colorsys
import numpy as np


def to_int_color(color):
    return tuple(int(x) for x in color)


def rgb_to_hsv(r, g, b):
    return list(colorsys.rgb_to_hsv(r, g, b))


def hsv_to_rgb(h, s, v):
    return list(colorsys.hsv_to_rgb(h, s, v))


def find_median_rgb(x1, x2):
    hsv_x1 = rgb_to_hsv(x1[0], x1[1], x1[2])
    hsv_x2 = rgb_to_hsv(x2[0], x2[1], x2[2])
    mean = (hsv_x1[0] + hsv_x2[1]) / 2
    res1 = hsv_x1.copy()
    res1[0] = mean
    res2 = hsv_x2.copy()
    res2[0] = mean
    res3 = hsv_x1.copy()
    res3[0] = 1 - mean
    res4 = hsv_x2.copy()
    res4[0] = 1 - mean
    return vals_to_rgb([res1, res2, res3, res4])


def vals_to_rgb(lst):
    return [hsv_to_rgb(x[0], x[1], x[2]) for x in lst]


def palette_to_triad_palette(predicted_palette, base_colors_num=3):
    predicted_palette = predicted_palette[:, :base_colors_num]
    btch_new_palette = []
    for btch_ind, pal in enumerate(predicted_palette):
        new_colors = []
        new_colors.extend(pal.copy())
        for i, x in enumerate(pal):
            for j in range(i + 1, len(pal)):
                c = find_median_rgb(x, pal[j])
                new_colors.extend(c)
        btch_new_palette.append(new_colors)
    return np.array(btch_new_palette)


def contrast_color_old(r, g, b):
    h, s, v = rgb_to_hsv(r, g, b)
    new_h = 0.5 + h if h < 0.5 else h - 0.5
    new_v = 50 + v if v < 50 else v - 50
    # new_h = 1 - h
    return hsv_to_rgb(new_h, 0, new_v)


def contrast_color(r, g, b):
    from utils.color_contrast import sufficient_contrast, contrast

    background_color = (r, g, b)
    i_r, i_g, i_b = caption_color = invert_color(r, g, b)
    if sufficient_contrast(caption_color, background_color):
        return caption_color
    # logger.warning("CAPT: Insufficient contrast, fixing")
    black = (10, 10, 10)
    white = (230, 230, 230)
    black_contrast = contrast(background_color, black)
    white_contrast = contrast(background_color, white)
    return black if black_contrast > white_contrast else white


def invert_color(r, g, b):
    return 255 - r, 255 - g, 255 - b


if __name__ == '__main__':
    print(find_median_rgb((0, 255 / 255, 69 / 255), (0.1, 0.5, 0.3)))
