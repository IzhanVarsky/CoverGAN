import torch

from outer.colors_tools import palette_to_triad_palette


def colorize(paths, colors, use_triad=False, need_stroke=False):
    if use_triad:
        colors = colors.reshape(colors.shape[0], -1, 3)
        colors = palette_to_triad_palette(colors)
    background_color = colors[0]
    color_ind = 1
    for path in paths:
        if path["fill_color"] is not None:
            # was alpha channel
            path["fill_color"] = torch.cat((colors[color_ind], path["fill_color"]), dim=0)
        else:
            path["fill_color"] = colors[color_ind]
        color_ind += 1
        if need_stroke:
            if path["stroke_color"] is not None:
                # was alpha channel
                path["stroke_color"] = torch.cat((colors[color_ind], path["stroke_color"]), dim=0)
            else:
                path["stroke_color"] = colors[color_ind]
            color_ind += 1
        else:
            path["stroke_color"] = path["fill_color"]
    return background_color