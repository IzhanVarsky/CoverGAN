import os

import pydiffvg

from outer.svg_tools.svg_tools import *


def create_svg(f_name, centerX, centerY, radiusX, radiusY, angle):
    if centerY is None:
        centerY = centerX
    path_depth = 2
    canvas_size = 512
    # points = create_rect(torch.tensor([centerX, centerY]), torch.tensor([radiusX, radiusY]),
    #                      torch.tensor([angle]), path_depth)
    points = create_regular_fig(7, torch.tensor([centerX, centerY]), torch.tensor([radiusX]),
                                torch.tensor([angle]), path_depth)
    points = points.view(-1, 2)
    segment_count = len(points) // 3  # side_cnt * (2 ** (path_depth - 1))
    shapes = [pydiffvg.Rect(
        p_min=torch.Tensor([0.0, 0.0]),
        p_max=torch.Tensor([1.0, 1.0]) * canvas_size,
        stroke_width=torch.Tensor([0.0])
    )]
    shape_groups = [pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.FloatTensor([1, 1, 1, 1])
    )]
    stroke_width = torch.tensor(1)
    control_per_segment = 2  # 3 points = 1 base + 2 control
    num_control_points = torch.full((segment_count,), control_per_segment, dtype=torch.int32)
    path = pydiffvg.Path(num_control_points=num_control_points,
                         points=points,
                         stroke_width=stroke_width,
                         is_closed=False)
    shapes.append(path)
    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([1]),
                                     fill_color=torch.tensor([0.3, 0.6, 0.3, 1.0]))
    shape_groups.append(path_group)
    pydiffvg.save_svg(f_name, canvas_size, canvas_size, shapes, shape_groups)


folder = "./res_rect"
os.makedirs(folder, exist_ok=True)
for i, x in enumerate([[100, 100, 10, 30, 0 * (math.pi / 180)],
                       [200, 100, 100, 50, 10 * (math.pi / 180)],
                       ]):
    create_svg(f"{folder}/{i}.svg", *x)
