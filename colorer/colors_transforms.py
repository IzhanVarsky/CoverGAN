import numpy as np
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color


def rgb_to_cielab(a):
    a = np.array(a)
    a1, a2, a3 = a / 255
    color1_rgb = sRGBColor(a1, a2, a3)
    color1_lab = convert_color(color1_rgb, LabColor)
    return np.array([color1_lab.lab_l, color1_lab.lab_a, color1_lab.lab_b])


def cielab_rgb_to(a):
    a = np.array(a)
    a1, a2, a3 = a
    lab = LabColor(a1, a2, a3)
    color = convert_color(lab, sRGBColor)
    return np.array([color.rgb_r, color.rgb_g, color.rgb_b])


def rgb_lab_rgb(a):
    return list((cielab_rgb_to(rgb_to_cielab(a)) * 255).astype(int))
