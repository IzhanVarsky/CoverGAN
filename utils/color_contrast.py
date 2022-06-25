from typing import Tuple


def luminance(rgb: Tuple[int, int, int]):
    # https://www.w3.org/TR/2008/REC-WCAG20-20081211/#relativeluminancedef
    def f(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4

    r, g, b = rgb
    return f(r) * 0.2126 + f(g) * 0.7152 + f(b) * 0.0722


def contrast(rgb1: Tuple[int, int, int], rgb2: Tuple[int, int, int]) -> float:
    # https://www.w3.org/TR/2008/REC-WCAG20-20081211/#contrast-ratiodef
    lum1 = luminance(rgb1)
    lum2 = luminance(rgb2)
    brightest = max(lum1, lum2)
    darkest = min(lum1, lum2)
    return (brightest + 0.05) / (darkest + 0.05)


def sufficient_contrast(rgb1: Tuple[int, int, int],
                        rgb2: Tuple[int, int, int]) -> bool:
    # https://www.w3.org/TR/2008/REC-WCAG20-20081211/#visual-audio-contrast-contrast
    return contrast(rgb1, rgb2) >= 3  # Consider 4.5 for smaller text
