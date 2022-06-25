import math
import random
from enum import Enum
from typing import List

import torch

PI = math.pi


def midpoints(pointList):
    def midpoint(p, q):
        return (p + q) / 2.0

    return [midpoint(pointList[i], pointList[i + 1]) for i in range(len(pointList) - 1)]


def subdivide(curve: List[torch.Tensor]):
    firstMidpoints = midpoints(curve)
    secondMidpoints = midpoints(firstMidpoints)
    thirdMidpoints = midpoints(secondMidpoints)
    a = [curve[0], firstMidpoints[0], secondMidpoints[0], thirdMidpoints[0]]
    b = [thirdMidpoints[0], secondMidpoints[1], firstMidpoints[2], curve[3]]
    return [a, b]


def cut_paths(paths: List[List[torch.Tensor]], max_depth: int, cur_depth=1):
    if cur_depth == max_depth:
        # return [torch.cat(path[:-1]) for path in paths]
        return [item for path in paths for item in path[:-1]]

    new_paths = [item for path in paths for item in subdivide(curve=path)]
    return cut_paths(new_paths, max_depth, cur_depth + 1)


def get_all_path_segment_count(start_path_segment_count: int, path_depth: int):
    return start_path_segment_count * (2 ** (path_depth - 1))


def rotate_fig_from_zero_point(figure: torch.Tensor, angle: torch.Tensor):
    all_trans = torch.cat((torch.cos(angle), -torch.sin(angle), torch.sin(angle), torch.cos(angle)), dim=-1).view(-1, 2)
    all_trans = all_trans.to(figure.device)
    return figure.view(-1, 2).matmul(all_trans)


def rotate_fig(figure: torch.Tensor, center: torch.Tensor, angle: torch.Tensor):
    figure = figure.view(-1, 2)
    figure = figure - center
    res = rotate_fig_from_zero_point(figure, angle)
    res = center + res
    return res


def make_straight_line(a1: torch.Tensor, a2: torch.Tensor):
    step = (a2 - a1) / 3
    return [a1, a1 + step, a2 - step, a2]


def fig_from_control_points(points):
    return [make_straight_line(p, points[(i + 1) % len(points)]) for i, p in enumerate(points)]


def enclose_figure(fig):
    fig = fig.view(-1, 2)
    return torch.cat((fig, fig[0]))


def deform_and_rotate(fig: torch.Tensor, center: torch.Tensor, rotationAngle: torch.Tensor,
                      deformation: torch.Tensor = None):
    if deformation is not None:
        fig = fig.flatten() + deformation
    return rotate_fig(fig, center, rotationAngle).flatten()


def create_circle_control_points(center, radius, segment_count):
    device = radius.device
    points = []
    a = 4 / 3 * math.tan(PI / 2 / segment_count) * radius
    bezier_rad = torch.sqrt(radius * radius + a ** 2).to(device)
    bezier_angle = torch.arctan(a / radius).to(device)
    angles = [ind * (2 * PI / segment_count) for ind in range(segment_count + 1)]
    for ind in range(segment_count):
        cur_angle = angles[ind]
        next_angle = angles[ind + 1]
        points.append(center + torch.tensor([math.cos(cur_angle) * radius, math.sin(cur_angle) * radius]).to(device))
        points.append(center + torch.tensor([torch.cos(cur_angle + bezier_angle).to(device) * bezier_rad,
                                             torch.sin(cur_angle + bezier_angle).to(device) * bezier_rad]).to(device))
        points.append(center + torch.tensor([torch.cos(next_angle - bezier_angle).to(device) * bezier_rad,
                                             torch.sin(next_angle - bezier_angle).to(device) * bezier_rad]).to(device))
    # points.append(points[0])  # to close the path
    return torch.cat(points, dim=-1).float().to(device)


def create_oval_base_vers(center: torch.Tensor, radius: torch.Tensor, rotationAngle: torch.Tensor, path_depth: int,
                          deformation: None,
                          deformation_coef_top_r: float = 1,
                          deformation_coef_top_l: float = 1,
                          deformation_coef_bottom_r: float = 1,
                          deformation_coef_bottom_l: float = 1,
                          ):
    radiusX, radiusY = radius
    width_two_thirds = radiusX * 4 / 3
    d1 = torch.cat((-torch.sin(rotationAngle).to(center.device),
                    torch.cos(rotationAngle).to(center.device))) * radiusY
    d2 = torch.cat((torch.cos(rotationAngle).to(center.device),
                    torch.sin(rotationAngle).to(center.device))) * width_two_thirds

    topCenter = (center + d1)
    topRight = (topCenter + d2) * deformation_coef_top_r
    topLeft = (topCenter - d2) * deformation_coef_top_l

    bottomCenter = (center - d1)
    bottomRight = (bottomCenter + d2) * deformation_coef_bottom_r
    bottomLeft = (bottomCenter - d2) * deformation_coef_bottom_l

    points = cut_paths([[bottomCenter, bottomRight, topRight, topCenter],
                        [topCenter, topLeft, bottomLeft, bottomCenter]], path_depth)
    return torch.cat(points).flatten()


def create_oval(center: torch.Tensor, radius: torch.Tensor, rotationAngle: torch.Tensor,
                path_depth: int, deformation: torch.Tensor = None):
    radiusX, radiusY = radius
    width_two_thirds = radiusX * 4 / 3
    d1 = torch.tensor([0, radiusY]).to(center.device)
    d2 = torch.tensor([width_two_thirds, 0]).to(center.device)

    topCenter = center + d1
    topRight = topCenter + d2
    topLeft = topCenter - d2

    bottomCenter = center - d1
    bottomRight = bottomCenter + d2
    bottomLeft = bottomCenter - d2

    points = cut_paths([[bottomCenter, bottomRight, topRight, topCenter],
                        [topCenter, topLeft, bottomLeft, bottomCenter]], path_depth)
    points = torch.cat(points)
    return deform_and_rotate(points, center, rotationAngle, deformation)


def create_circle2(center: torch.Tensor, radius: torch.Tensor, rotationAngle: torch.Tensor,
                   path_depth: int, deformation: torch.Tensor = None):
    return create_oval(center, torch.cat((radius, radius)), rotationAngle, path_depth, deformation)


def create_rect(center: torch.Tensor, radius: torch.Tensor, rotationAngle: torch.Tensor,
                path_depth: int, deformation: torch.Tensor = None):
    changer = torch.tensor([1, -1]).to(center.device)
    plus_minus_rad = radius * changer
    a1 = -radius
    a2 = plus_minus_rad
    a3 = radius
    a4 = -plus_minus_rad

    control_points = fig_from_control_points([a1, a2, a3, a4])
    points = cut_paths(control_points, path_depth)
    points = torch.cat(points).view(-1, 2) + center
    return deform_and_rotate(points, center, rotationAngle, deformation)


def create_regular_fig(side_count: int, center: torch.Tensor, big_rad: torch.Tensor, rotationAngle: torch.Tensor,
                       path_depth: int, deformation: torch.Tensor = None):
    points = []
    angles = [ind * (2 * PI / side_count) for ind in range(side_count)]
    for cur_angle in angles:
        points.append(center + torch.tensor([math.cos(cur_angle) * big_rad,
                                             math.sin(cur_angle) * big_rad]).to(center.device))
    points = fig_from_control_points(points)
    points = cut_paths(points, path_depth)
    points = torch.cat(points)
    return deform_and_rotate(points, center, rotationAngle, deformation)


def create_triangle(center: torch.Tensor, big_rad: torch.Tensor, rotationAngle: torch.Tensor,
                    path_depth: int, deformation: torch.Tensor = None):
    return create_regular_fig(3, center, big_rad, rotationAngle, path_depth, deformation)


def create_square(center: torch.Tensor, big_rad: torch.Tensor, rotationAngle: torch.Tensor,
                  path_depth: int, deformation: torch.Tensor = None):
    return create_regular_fig(4, center, big_rad, rotationAngle, path_depth, deformation)


def create_pentagon(center: torch.Tensor, big_rad: torch.Tensor, rotationAngle: torch.Tensor,
                    path_depth: int, deformation: torch.Tensor = None):
    return create_regular_fig(5, center, big_rad, rotationAngle, path_depth, deformation)


def create_rand_fig(center: torch.Tensor, big_rad: torch.Tensor, rotationAngle: torch.Tensor,
                    path_depth: int, deformation: torch.Tensor = None):
    rand_func = random.choice([init_func_types_config[x][0]
                               for x in init_func_types_config
                               if init_func_types_config[x] is not InitFuncType.RAND])
    return rand_func(center, big_rad, rotationAngle, path_depth, deformation)


def add_center_fig(shapes, shape_groups, center, circle_rad=3.0, color=[1, 0, 1, 1.0]):
    import pydiffvg
    path = pydiffvg.Circle(radius=torch.tensor(circle_rad), center=torch.tensor(center))
    shapes.append(path)
    path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shape_groups)]),
                                     fill_color=torch.tensor(color))
    shape_groups.append(path_group)


def add_points_as_circles(shapes, shape_groups, points, circle_rad=3.0,
                          color1=[0.0, 0.0, 0.0, 1.0],
                          color2=[1, 0, 0, 1.0]):
    for idx, center in enumerate(points):
        fill_color = color1 if idx % 3 == 0 else color2
        add_center_fig(shapes, shape_groups, center, circle_rad=circle_rad, color=fill_color)


class InitFuncType(Enum):
    CIRCLE = 0,
    OVAL = 1,
    RECT = 2,
    TRIANGLE = 3,
    SQUARE = 4,
    PENTAGON = 5,
    RAND = 6,


class BaseMethodConfig:
    def __init__(self, function, start_fig_segment_count, radius_count):
        self.base_function = function
        self.start_fig_segment_count = start_fig_segment_count
        self.radius_count = radius_count


init_func_types_config = {
    InitFuncType.CIRCLE: BaseMethodConfig(create_circle2, 2, 1),
    InitFuncType.OVAL: BaseMethodConfig(create_oval, 2, 2),
    InitFuncType.RECT: BaseMethodConfig(create_rect, 4, 2),
    InitFuncType.TRIANGLE: BaseMethodConfig(create_triangle, 3, 1),
    InitFuncType.SQUARE: BaseMethodConfig(create_square, 4, 1),
    InitFuncType.PENTAGON: BaseMethodConfig(create_pentagon, 5, 1),
    InitFuncType.RAND: BaseMethodConfig(create_rand_fig, 5, 2),
}

fig_classes = {
    0: init_func_types_config[InitFuncType.CIRCLE],
    1: init_func_types_config[InitFuncType.OVAL],
    2: init_func_types_config[InitFuncType.RECT],
    3: init_func_types_config[InitFuncType.TRIANGLE],
    4: init_func_types_config[InitFuncType.SQUARE],
    5: init_func_types_config[InitFuncType.PENTAGON],
}
