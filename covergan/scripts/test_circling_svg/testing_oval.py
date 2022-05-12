import math
import os

import pydiffvg
import torch

from outer.svg_tools.svg_tools import create_oval


def create_svg(f_name, centerX, centerY, radiusX, radiusY, angle):
    if centerY is None:
        centerY = centerX
    path_depth = 1
    segment_count = 2 ** path_depth
    canvas_size = 256
    points = create_oval(torch.tensor([centerX, centerY]), torch.tensor([radiusX, radiusY]),
                         torch.tensor([angle]), path_depth,
                         # deformation_coef_top_r=1.5,
                         # deformation_coef_top_l=0.5,
                         # deformation_coef_bottom_r=0.75,
                         # deformation_coef_bottom_l=0.75,
                         ).view(-1, 2)
    points = (points * 10).round().to(int) // 10
    shapes = [pydiffvg.Rect(
        p_min=torch.Tensor([0.0, 0.0]),
        p_max=torch.Tensor([1.0, 1.0]) * canvas_size,
        stroke_width=torch.Tensor([0.0])
    )]
    shape_groups = [pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([1, 1, 1, 1.0]).float()
    )]
    stroke_width = torch.tensor(1)
    control_per_segment = 2  # 3 points = 1 base + 2 control
    num_control_points = torch.full((segment_count,), control_per_segment, dtype=torch.int32)
    path = pydiffvg.Path(num_control_points=num_control_points,
                         points=points,
                         stroke_width=stroke_width,
                         is_closed=True)
    shapes.append(path)
    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([1]),
                                     fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0]))
    shape_groups.append(path_group)
    pydiffvg.save_svg(f_name, canvas_size, canvas_size, shapes, shape_groups)


folder = "./res_oval"
os.makedirs(folder, exist_ok=True)
for i, x in enumerate([[100, 100, 10, 30, 180],
                       [128, 128, 100, 50, 0 * (math.pi / 180)],
                       [128, 128, 30, 30, 0 * (math.pi / 180)],
                       ]):
    create_svg(f"{folder}/{i}.svg", *x)
