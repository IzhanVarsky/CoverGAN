from colorer.music_palette_dataset import get_main_rgb_palette, get_main_rgb_palette2
from lxml import etree as ET
from enum import Enum


class TransferAlgoType(Enum):
    ColorThiefUnsorted = 1
    UnsortedClustering = 2
    SortedClustering = 3


def transfer_style(svg_path, png_path, svg_out_path, algo_type: TransferAlgoType = TransferAlgoType.SortedClustering):
    print(f"Transferring style from `{png_path}` to svg file `{svg_path}`. Out path: `{svg_out_path}`")
    tree = ET.parse(svg_path)
    root = tree.getroot()
    nodes = root.findall("*[@fill]")
    if algo_type == TransferAlgoType.ColorThiefUnsorted:
        palette = get_main_rgb_palette(png_path, color_count=len(nodes))
    elif algo_type == TransferAlgoType.UnsortedClustering:
        palette = get_main_rgb_palette2(png_path, color_count=len(nodes), sort_colors=False)
    else:
        palette = get_main_rgb_palette2(png_path, color_count=len(nodes), sort_colors=True)
    for ind, n in enumerate(nodes):
        r, g, b = palette[ind % len(palette)]
        n.set("fill", f"rgb({r}, {g}, {b})")
    tree.write(svg_out_path, encoding="utf-8")


if __name__ == '__main__':
    transfer_style("test_svg.svg", "A S T R O - Change.jpg", "out.svg")
    transfer_style("test_svg2.svg", "img.png", "out2.svg")
    transfer_style("img3.svg", "img.png", "out3.svg")
