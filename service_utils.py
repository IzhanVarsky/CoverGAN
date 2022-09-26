import math
import os
import random
from enum import Enum
from io import BytesIO
from typing import Tuple, Optional

import torch
from PIL import Image, ImageFont
from kornia.color import rgb_to_rgba
from torchvision.transforms.functional import to_tensor

from outer.SVGContainer import *
from outer.represent import color_to_rgba_attr
from utils.bboxes import BBox, merge_bboxes
from utils.color_contrast import sufficient_contrast, contrast
from utils.color_extractor import extract_primary_color
from utils.text_fitter import fit_text

SERVICE_NAME = "PROTECTED"


class OverlayFilter(Enum):
    DIM = 0
    VIGNETTE = 1
    VINTAGE = 2
    SEPIA = 3
    WHITEN = 4


def color_to_rgb_attr(color):
    return f"rgb({color[0]}, {color[1]}, {color[2]})"


def add_filter(image: SVGContainer, f: OverlayFilter):
    rect = RectTag(attrs_dict={"x": 0, "y": 0,
                               "width": image.width,
                               "height": image.height,
                               })
    image.add_inner_node(rect)

    if f == OverlayFilter.DIM:
        rect.add_attr("fill", color_to_rgba_attr([0, 0, 0, 100]))
    elif f == OverlayFilter.VINTAGE:
        rect.add_attr("fill", color_to_rgba_attr([255, 244, 226, 100]))
    elif f == OverlayFilter.SEPIA:
        rect.add_attr("fill", color_to_rgba_attr([173, 129, 66, 100]))
    elif f == OverlayFilter.WHITEN:
        rect.add_attr("fill", color_to_rgba_attr([255, 255, 255, 100]))
    elif f == OverlayFilter.VIGNETTE:
        radial_gradient = RadialGradientTag()
        radial_gradient.add_stop(0, color_to_rgba_attr([0, 0, 0, 0]))
        radial_gradient.add_stop(0.7, color_to_rgba_attr([0, 0, 0, 0]))
        radial_gradient.add_stop(1, color_to_rgba_attr([0, 0, 0, 153]))
        radial_gradient.add_attrs({"cx": image.width // 2,
                                   "cy": image.height // 2,
                                   "r": math.ceil(image.width / math.sqrt(2)),
                                   "gradientUnits": "userSpaceOnUse",
                                   "spreadMethod": "pad"})
        image.bind_tags_with_id(rect, radial_gradient, "fill")
        image.insert_inner_node(1, radial_gradient)


def add_watermark(image: SVGContainer):
    text_tag = TextTag(SERVICE_NAME, attrs_dict={"x": image.width - 300,
                                                 "y": image.height - 30,
                                                 "font-family": "Roboto",
                                                 "font-weight": 700,
                                                 "font-size": 40,
                                                 "writing-mode": "lr",
                                                 "fill": color_to_rgba_attr([255, 255, 255, 100]),
                                                 })
    image.add_inner_node(text_tag)
    image.font_importer.add_font("Roboto")


def pos_tensor_to_bbox(t: torch.Tensor) -> BBox:
    x1 = t[0].item()
    y1 = t[1].item()
    w = t[2].item()
    h = t[3].item()
    return BBox(x1, y1, x1 + w, y1 + h)


def png_data_to_pil_image(png_data, canvas_size: Optional[int] = None) -> Image:
    result = Image.open(BytesIO(png_data)).convert('RGB')
    if canvas_size is not None:
        result = result.resize((canvas_size, canvas_size))
    return result


def get_region_color(img: Image.Image, bbox: BBox) -> Optional[Tuple[int, int, int]]:
    region = img.crop((bbox.x1, bbox.y1, bbox.x2, bbox.y2))
    region = rgb_to_rgba(to_tensor(region), 1.0).permute(1, 2, 0)  # CHW -> HWC
    primary_color = extract_primary_color(region, count=3)
    return primary_color


def ensure_caption_contrast(img: Image.Image,
                            region: BBox,
                            caption_color: Tuple[int, int, int]) -> Tuple[int, int, int]:
    background_color = get_region_color(img, region)
    if background_color is None:
        return caption_color
    if sufficient_contrast(caption_color, background_color):
        return caption_color
    else:
        # logger.warning("CAPT: Insufficient contrast, fixing")
        black = (10, 10, 10)
        white = (230, 230, 230)
        black_contrast = contrast(background_color, black)
        white_contrast = contrast(background_color, white)
        return black if black_contrast > white_contrast else white


def get_font_filename(font_dir: str, font_family: str, bold: bool):
    font_family = font_family.replace(' ', '')
    bold_filename = f"{font_dir}/{font_family}-Bold.ttf"
    regular_filename = f"{font_dir}/{font_family}-Regular.ttf"
    if bold:
        return bold_filename if os.path.isfile(bold_filename) else regular_filename
    else:
        return regular_filename


def add_ready_text_node(image: SVGContainer,
                        text: str,
                        text_color: tuple,
                        font: ImageFont.FreeTypeFont,
                        xy, rt,
                        debug=False):
    x, y = xy
    r, t = rt
    textLength = r - x
    font_family = font.font.family
    font_style = font.font.style
    end_font_stretch = "normal"
    end_font_style = "normal"
    if "SemiCondensed" in font_family:
        end_font_stretch = "semi-condensed"
        font_family = font_family.replace("SemiCondensed", "").strip()
    elif "Condensed" in font_family:
        end_font_stretch = "condensed"
        font_family = font_family.replace("Condensed", "").strip()
    if "Italic" in font_style:
        end_font_style = "italic"
        font_style = font_style.replace("Italic", "").strip()
    font_weight = 400
    if font_style == "Thin":
        font_weight = 100
    elif font_style == "ExtraLight":
        font_weight = 200
    elif font_style == "Light":
        font_weight = 300
    elif font_style == "Regular":
        font_weight = 400
    elif font_style == "Medium":
        font_weight = 500
    elif font_style == "SemiBold":
        font_weight = 600
    elif font_style == "Bold":
        font_weight = 700
    elif font_style == "ExtraBold":
        font_weight = 800
    elif font_style == "Black":
        font_weight = 900
    elif font.font.style != "Italic":
        print(f"Unexpected font style: `{font_style}`")
    text_tag = TextTag(text, attrs_dict={"x": x, "y": y,
                                         "font-family": font_family,
                                         "font-weight": font_weight,
                                         "font-stretch": end_font_stretch,
                                         "font-size": font.size,
                                         "textLength": textLength,
                                         "font-style": end_font_style,
                                         "writing-mode": "lr",
                                         "fill": color_to_rgba_attr([*text_color, 255]),
                                         })
    image.add_inner_node(text_tag)
    image.font_importer.add_font(font_family)
    if debug:
        image.add_inner_node(CircleTag.create(2, x, y, "green"))


def make_text_node(image: SVGContainer,
                   text: str,
                   text_pos: BBox,
                   text_color: tuple,
                   text_font_family: str,
                   font_dir: str,
                   debug: bool = False):
    font_bold = bool(random.getrandbits(1))
    font_weight = 700 if font_bold else 400
    font_filename = get_font_filename(font_dir, text_font_family, font_bold)

    if '\n' in text:
        part1, part2 = text.split('\n')
        bbox_splits = [
            text_pos.split_horizontal(0.3),
            text_pos.split_horizontal(0.5),
            text_pos.split_horizontal(0.7),
            text_pos.split_vertical(0.3),
            text_pos.split_vertical(0.5),
            text_pos.split_vertical(0.7),
        ]

        best_score = 0
        best_split = None
        for (bbox1, bbox2) in bbox_splits:
            score1 = fit_text(part1, bbox1, font_filename)[-1]
            score2 = fit_text(part2, bbox2, font_filename)[-1]
            score = (score1 * bbox1.area() + score2 * bbox2.area()) / text_pos.area()
            if score >= best_score:
                best_split = (bbox1, bbox2)

        bbox1, bbox2 = best_split
        make_text_node(image, part1, bbox1, text_color, text_font_family, font_dir, debug)
        make_text_node(image, part2, bbox2, text_color, text_font_family, font_dir, debug)
        return

    font_size, direction, x, y, score = fit_text(text, text_pos, font_filename)

    if not score:
        # logger.warning("FIT: Failed to fit the text!")
        pass

    writingMode = "tb" if direction == 'ttb' else "lr"
    lengthAdjust = None
    textLength = None

    set_text_length = bool(random.getrandbits(1))
    if set_text_length:
        if direction == 'ttb':
            y = text_pos.y1
            textLength = text_pos.height()
        else:
            x = text_pos.x1
            textLength = text_pos.width()
        adjust_glyphs = bool(random.getrandbits(1))
        if adjust_glyphs:
            lengthAdjust = "spacingAndGlyphs"

    attrs_dict = {"x": x, "y": y,
                  "font-family": text_font_family,
                  "font-weight": font_weight,
                  "font-size": font_size,
                  "writing-mode": writingMode,
                  "fill": color_to_rgba_attr([*text_color, 255]),
                  }
    if lengthAdjust is not None:
        attrs_dict["lengthAdjust"] = lengthAdjust
    if textLength is not None:
        attrs_dict["textLength"] = textLength

    text_tag = TextTag(text, attrs_dict=attrs_dict)
    image.add_inner_node(text_tag)
    image.font_importer.add_font(text_font_family)

    if debug:
        attrs_dict = {"x": text_pos.x1, "y": text_pos.y1,
                      "width": text_pos.width(),
                      "height": text_pos.height(),
                      "fill": color_to_rgba_attr([0, 0, 0, 0]),
                      "stroke": color_to_rgba_attr([255, 255, 255, 255]),
                      }
        rect = RectTag(attrs_dict=attrs_dict)
        image.add_inner_node(rect)


def add_caption(psvg_cover: SVGContainer, font_dir: str,
                track_artist: str, artist_name_pos: BBox, artist_name_color: tuple,
                track_name: str, track_name_pos: BBox, track_name_color: tuple,
                debug: bool, deterministic: bool = False):
    # Open fonts available from https://fonts.google.com
    FONTS = [
        "Allerta Stencil",
        "Balsamiq Sans",
        "Cabin",
        "Caveat",
        "Comfortaa",
        "Fira Sans",
        "Lobster",
        "Montserrat",
        "Neucha",
        "Nunito",
        "Open Sans",
        "Oxanium",
        "Playfair Display",
        "Press Start 2P",
        "PT Serif",
        "Reggae One",
        "Roboto",
        "Schoolbell",
        "Special Elite",
        "Titillium Web",
        "Zilla Slab",
    ]
    artist_font_family = FONTS[0] if deterministic else random.choice(FONTS)
    name_font_family = FONTS[0] if deterministic else random.choice(FONTS)

    if artist_name_pos.overlaps(track_name_pos):
        merged_pos = merge_bboxes([artist_name_pos, track_name_pos])
        merged_color = artist_name_color
        merged_font_family = artist_font_family
        merged_text = '\n'.join([track_artist, track_name])
        make_text_node(psvg_cover, merged_text, merged_pos, merged_color, merged_font_family, font_dir, debug)
    else:
        make_text_node(psvg_cover, track_artist, artist_name_pos, artist_name_color,
                       artist_font_family, font_dir, debug)
        make_text_node(psvg_cover, track_name, track_name_pos, track_name_color, name_font_family, font_dir, debug)


def paste_caption(svg_cont: SVGContainer,
                  pil_img,
                  track_artist: str,
                  track_name: str,
                  font_dir: str,
                  for_svg=True):
    from PIL import ImageDraw
    from utils.deterministic_text_fitter import get_all_boxes_info_to_paste
    import numpy as np
    from utils.deterministic_text_fitter import draw_to_draw_object
    debug = False
    draw = ImageDraw.Draw(pil_img, mode='RGB')
    to_draw = get_all_boxes_info_to_paste(track_artist, track_name, np.asarray(pil_img), font_dir,
                                          for_svg=for_svg)
    for x in to_draw:
        draw_to_draw_object(draw, x)
        if x["type"] == "text":
            l, t = x["text_xy_left_top"]
            ascent = x["ascent"]
            t += ascent - 1
            r, b = x["text_xy_right_bottom"]
            b -= 10
            add_ready_text_node(svg_cont, x["word"], x["color"], x["font"], (l, b), (r, t), debug=debug)
            if debug:
                svg_cont.add_inner_node(CircleTag.create(2, l, b + 10, "lightgreen"))
        elif debug and x["type"] == "circle":
            d = x
            p_y, p_x = d["xy"]
            r = d["r"]
            descent = d["descent"]
            ascent = d["ascent"]
            y_t, x_l = d["text_xy_left_top"]
            # draw_circle(draw, x_l + ascent, y_t, 1, d["color"])
            svg_cont.add_inner_node(CircleTag.create(2, y_t, x_l, "yellow"))
            svg_cont.add_inner_node(CircleTag.create(2, y_t, x_l + descent, "black"))
            svg_cont.add_inner_node(CircleTag.create(2, y_t, x_l + ascent,
                                                     color_to_rgb_attr(d["color"])))
            svg_cont.add_inner_node(CircleTag.create(2, y_t, x_l + ascent + descent, "pink"))

def add_font_imports_to_svg_xml(svg_xml):
    from xml.dom import minidom
    tree = minidom.parseString(svg_xml)
    texts = tree.getElementsByTagName("text")
    families = set()
    for t in texts:
        style = t.attributes['style'].value
        attrs = style.split(";")
        font_attrs = list(filter(lambda x: x.startswith("font-family"), attrs))
        if len(font_attrs) != 0:
            families.add(font_attrs[0].split(":")[1])
    style_text = ""
    for f in families:
        style_text += f"@import url('https://fonts.googleapis.com/css?family={f}');\n"
    split = svg_xml.split("\n")
    split.insert(1, f"<style>{style_text}</style>")
    return "\n".join(split)
