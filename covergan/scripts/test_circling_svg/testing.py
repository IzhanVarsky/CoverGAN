import os

import pydiffvg
import torch

from outer.svg_tools.svg_tools import create_circle_control_points, cut_paths


def create_circle_points(center, radius):
    # center = (200, 350)
    # radius = 50
    # <path fill="gray" d="M 200,300
    #              C 150,300 150,350 150,350
    #              C 150,350 150,400 200,400
    #              C 250,400 250,350 250,350
    #              C 250,300 200,300 200,300"/>
    #     <circle cx="200" cy="300" r="5" fill="red"/>
    #     <circle cx="150" cy="350" r="5" fill="yellow"/>
    #     <circle cx="200" cy="400" r="5" fill="blue"/>
    #     <circle cx="250" cy="350" r="5" fill="pink"/>
    #     <circle cx="200" cy="350" r="5" fill="purple"/>
    zero_rad = torch.cat((torch.tensor([0]).to(radius.device), radius), dim=0)
    rad_zero = torch.cat((radius, torch.tensor([0]).to(radius.device)), dim=0)
    p1 = center - zero_rad
    p2 = p1 - rad_zero
    p3 = p2 + zero_rad
    p4 = p3
    p5 = p4
    p6 = p5 + zero_rad
    p7 = p6 + rad_zero
    p8 = p7 + rad_zero
    p9 = p8 - zero_rad
    p10 = p9
    p11 = p10 - zero_rad
    p12 = p11 - rad_zero
    p13 = p12
    return torch.cat((p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13), dim=0).float()


def create_svg(f_name, centerX, radius, centerY=None):
    if centerY is None:
        centerY = centerX
    segment_count = 8
    # center = 256, radius = 30
    canvas_size = 512
    # points = create_circle_points(torch.tensor([centerX, centerY]), torch.tensor([radius])).view(-1, 2)
    points = create_circle_control_points(torch.tensor([centerX, centerY]),
                                          torch.tensor([radius]), segment_count).view(-1, 2)
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


def create_circle_control_points_svg(f_name, centerX, radius, centerY=None, segment_count=2):
    if centerY is None:
        centerY = centerX
    canvas_size = 512
    points = create_circle_control_points(torch.tensor([centerX, centerY]),
                                          torch.tensor([radius]), segment_count).view(-1, 2)
    shapes = [pydiffvg.Rect(
        p_min=torch.Tensor([0.0, 0.0]),
        p_max=torch.Tensor([1.0, 1.0]) * canvas_size,
        stroke_width=torch.Tensor([0.0])
    )]
    shape_groups = [pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=torch.tensor([1, 1, 1, 1.0]).float()
    )]
    for idx, center in enumerate(points):
        path = pydiffvg.Circle(radius=torch.tensor(3.0), center=center)
        shapes.append(path)
        if idx % 3 == 0:
            fill_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
        else:
            fill_color = torch.tensor([0.3, 0.6, 0.3, 1.0])
        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shape_groups)]),
                                         fill_color=fill_color)
        shape_groups.append(path_group)
    pydiffvg.save_svg(f_name, canvas_size, canvas_size, shapes, shape_groups)


# paths = cut_paths([[torch.tensor([40, 50]),
#                     torch.tensor([60, 120]),
#                     torch.tensor([180, 130]),
#                     torch.tensor([200, 230])]], 2)
# 200,350 C 150,250 100,150 50,50
paths = cut_paths([[torch.tensor([200, 350]),
                    torch.tensor([150, 250]),
                    torch.tensor([100, 150]),
                    torch.tensor([50, 50])]], 2)
for p in paths:
    for x in p.detach().numpy():
        print(x, end=" ")

os.makedirs("./res", exist_ok=True)
for i, x in enumerate([[100, 10],
                       # [100, 50],
                       # [100, 70],
                       # [100, 100],
                       # [100, 110],
                       # [200, 10],
                       # [200, 50],
                       # [200, 100],
                       # [200, 200],
                       # [300, 200],
                       # [400, 200],
                       # [500, 200],
                       [200, 100, 300],
                       ]):
    if len(x) < 3:
        x.append(None)
    create_svg(f"./res/{i}.svg", x[0], x[1], x[2])
    # create_circle_control_points_svg(f"./res/{i}.svg", x[0], x[1], x[2])