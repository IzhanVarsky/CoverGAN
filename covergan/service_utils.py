import math
import os
import random
from enum import Enum
from io import BytesIO
from typing import Tuple, Optional

import protosvg.protosvg_pb2 as psvg
import torch
from PIL import Image, ImageFont
from kornia.color import rgb_to_rgba
from torchvision.transforms.functional import to_tensor

from protosvg.client import color_to_int
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


def add_filter(image: psvg.ProtoSVG, f: OverlayFilter):
    overlay = psvg.Square()
    overlay.start.x = 0
    overlay.start.y = 0
    overlay.width = image.width
    s = image.shapes.add()
    s.square.CopyFrom(overlay)

    if f == OverlayFilter.DIM:
        s.color.rgba = color_to_int(r=0, g=0, b=0, a=100)
    elif f == OverlayFilter.VINTAGE:
        s.color.rgba = color_to_int(r=255, g=244, b=226, a=100)
    elif f == OverlayFilter.SEPIA:
        s.color.rgba = color_to_int(r=173, g=129, b=66, a=100)
    elif f == OverlayFilter.WHITEN:
        s.color.rgba = color_to_int(r=255, g=255, b=255, a=100)
    elif f == OverlayFilter.VIGNETTE:
        gradient = psvg.RadialGradient()

        # Center
        stop = gradient.base.stops.add()
        stop.offset = 0
        stop.color.rgba = color_to_int(r=0, g=0, b=0, a=0)
        # Circular start
        stop = gradient.base.stops.add()
        stop.offset = 0.7
        stop.color.rgba = color_to_int(r=0, g=0, b=0, a=0)
        # End
        stop = gradient.base.stops.add()
        stop.offset = 1.0
        stop.color.rgba = color_to_int(r=0, g=0, b=0, a=153)

        gradient.end.x = image.width // 2
        gradient.end.y = image.height // 2
        gradient.endRadius = math.ceil(image.width / math.sqrt(2))
        s.radialGradient.CopyFrom(gradient)


def add_watermark(image: psvg.ProtoSVG):
    label = psvg.Label()

    label.font.size = 40
    label.font.family = "Roboto"
    label.font.weight = 700

    label.text = SERVICE_NAME
    label.start.x = image.width - 300
    label.start.y = image.height - 30

    s = image.shapes.add()
    s.label.CopyFrom(label)
    s.color.rgba = color_to_int(r=255, g=255, b=255, a=100)


def rgb_tuple_to_int(rgb: Tuple[int, int, int]) -> int:
    return color_to_int(*rgb, a=255)


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


def add_ready_text_node(image: psvg.ProtoSVG,
                        text: str,
                        text_color: tuple,
                        font: ImageFont.FreeTypeFont,
                        xy, rt):
    label = psvg.Label()
    x, y = xy
    r, t = rt
    label.textLength = r - x
    label.font.size = font.size
    font_family = font.font.family
    font_style = font.font.style
    if "SemiCondensed" in font_family:
        label.font.stretch = label.font.SEMI_CONDENSED
        font_family = font_family.replace("SemiCondensed", "").strip()
    elif "Condensed" in font_family:
        label.font.stretch = label.font.CONDENSED
        font_family = font_family.replace("Condensed", "").strip()
    if "Italic" in font_style:
        label.font.style = label.font.ITALIC
        font_style = font_style.replace("Italic", "").strip()
    if font_style == "Thin":
        label.font.weight = 100
    elif font_style == "ExtraLight":
        label.font.weight = 200
    elif font_style == "Light":
        label.font.weight = 300
    elif font_style == "Regular":
        label.font.weight = 400
    elif font_style == "Medium":
        label.font.weight = 500
    elif font_style == "SemiBold":
        label.font.weight = 600
    elif font_style == "Bold":
        label.font.weight = 700
    elif font_style == "ExtraBold":
        label.font.weight = 800
    elif font_style == "Black":
        label.font.weight = 900
    elif font.font.style != "Italic":
        print(f"Unexpected font style: `{font_style}`")
    label.font.family = font_family
    label.text = text
    label.start.x, label.start.y = x, y
    s = image.shapes.add()
    s.label.CopyFrom(label)
    s.color.rgba = rgb_tuple_to_int(text_color)


def make_text_node(image: psvg.ProtoSVG,
                   text: str,
                   text_pos: BBox,
                   text_color: int,
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

    label = psvg.Label()

    label.font.size = font_size
    label.font.family = text_font_family
    label.font.weight = font_weight

    label.text = text
    if direction == 'ttb':
        label.writingMode = random.choice((
            psvg.Label.WritingMode.VERTICAL_RIGHT_LEFT,
            psvg.Label.WritingMode.VERTICAL_LEFT_RIGHT
        ))

    set_text_length = bool(random.getrandbits(1))
    if set_text_length:
        if direction == 'ttb':
            label.start.x = x
            label.start.y = text_pos.y1
            label.textLength = text_pos.height()
        else:
            label.start.x = text_pos.x1
            label.start.y = y
            label.textLength = text_pos.width()
        adjust_glyphs = bool(random.getrandbits(1))
        if adjust_glyphs:
            label.lengthAdjust = psvg.Label.LengthAdjust.SPACING_AND_GLYPHS
    else:
        label.start.x = x
        label.start.y = y

    s = image.shapes.add()
    s.label.CopyFrom(label)
    s.color.rgba = text_color

    if debug:
        rect = psvg.Rectangle()
        rect.start.x = text_pos.x1
        rect.start.y = text_pos.y1
        rect.width = text_pos.width()
        rect.height = text_pos.height()
        s = image.shapes.add()
        s.rectangle.CopyFrom(rect)
        s.color.rgba = color_to_int(r=0, g=0, b=0, a=0)
        s.stroke.width = 2
        s.stroke.color.rgba = color_to_int(r=255, g=255, b=255, a=255)


def add_caption(psvg_cover: psvg.ProtoSVG, font_dir: str,
                track_artist: str, artist_name_pos: BBox, artist_name_color: int, artist_font_family: str,
                track_name: str, track_name_pos: BBox, track_name_color: int, name_font_family: str, debug: bool):
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
    # print(families)
    style_text = ""
    for f in families:
        style_text += f"@import url('https://fonts.googleapis.com/css?family={f}');\n"
    split = svg_xml.split("\n")
    split.insert(1, f"<style>{style_text}</style>")
    return "\n".join(split)
    # from xml.etree import ElementTree as ET
    # root = ET.fromstring(svg_xml)
    # style = root.makeelement('style', {})
    # style.text = style_text
    # root.insert(0, style)
    # return ET.tostring(root, encoding='unicode')
