from typing import List

import torch

from outer.colors_tools import palette_to_triad_palette


class CoverFigure:
    def __init__(self):
        self.points = None
        self.fill_color = None
        self.stroke_width = None
        self.stroke_color = None

        self.center_point = None
        self.radius = None
        self.deformation_points = None
        self.angle = None

    def to_dict(self):
        return {"points": self.points, "fill_color": self.fill_color,
                "stroke_width": self.stroke_width, "stroke_color": self.stroke_color}


class Cover:
    def __init__(self):
        self.background_color = None
        self.canvas_size = None
        self.figures: List[CoverFigure] = []

    def add_figure(self, fig: CoverFigure):
        self.figures.append(fig)

    def colorize_cover(self, colors, use_triad=False, need_stroke=False):
        if use_triad:
            device = colors.device
            colors = colors.detach().cpu().numpy()[None, :]
            colors = palette_to_triad_palette(colors)[0]
            colors = torch.from_numpy(colors).to(device)
        self.background_color = colors[0]
        color_ind = 1
        for path in self.figures:
            if path.fill_color is not None:
                # was alpha channel
                path.fill_color = torch.cat((colors[color_ind], path.fill_color[-1:]), dim=0)
            else:
                path.fill_color = colors[color_ind]
            color_ind += 1
            if need_stroke:
                if path.stroke_color is not None:
                    # was alpha channel
                    path.stroke_color = torch.cat((colors[color_ind], path.stroke_color[-1:]), dim=0)
                else:
                    path.stroke_color = colors[color_ind]
                color_ind += 1
            else:
                path.stroke_color = path.fill_color

    def to_background_and_paths(self):
        return self.background_color, list(map(CoverFigure.to_dict, self.figures))
