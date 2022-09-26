from pydiffvg import *

from outer.SVGContainer import SVGContainer, RectTag, PathTag
from outer.models.cover_classes import Cover, CoverFigure


def tensor_color_to_int(t: torch.Tensor, a: float = None):
    t = (t * 255).round().to(int)
    r = t[0].item()
    g = t[1].item()
    b = t[2].item()
    if a is None:
        a = t[3].item()
    else:
        a = round(255 * a)
    return [r, g, b, a]


def color_to_rgba_attr(color):
    return f"rgba({color[0]}, {color[1]}, {color[2]}, {round(color[3] / 255.0, 2)})"


def to_diffvg_svg_params(paths: [dict], background_color: torch.Tensor, canvas_size: int):
    shapes = []
    shape_groups = []

    # No transparency for the background
    background_color = torch.cat((background_color, torch.ones(1, device=background_color.device)))
    background_square = pydiffvg.Rect(
        p_min=torch.Tensor([0.0, 0.0]),
        p_max=torch.Tensor([1.0, 1.0]) * canvas_size,
        stroke_width=torch.Tensor([0.0])
    )
    shapes.append(background_square)
    background_group = pydiffvg.ShapeGroup(
        shape_ids=torch.tensor([0]),
        fill_color=background_color
    )
    shape_groups.append(background_group)

    # Paths
    for p in paths:
        if "is_circle" in p:
            radius = p["radius"]  # radius=torch.tensor(40.0),
            center = p["center"]  # center=torch.tensor([128.0, 128.0])
            stroke_color = p["stroke_color"]
            fill_color = p["fill_color"]
            path = pydiffvg.Circle(radius=radius, center=center)
        else:
            points = p["points"]  # For `segment_count` segments
            stroke_width = p["stroke_width"]
            stroke_color = p["stroke_color"]
            fill_color = p["fill_color"]

            # Avoid overlapping points
            # Note: this is taken from diffvg GAN example
            eps = 1e-4
            points = points + eps * torch.randn_like(points)

            control_per_segment = 2  # 3 points = 1 base + 2 control
            segment_count = (len(points) - 1) // 3
            num_control_points = torch.full((segment_count,), control_per_segment, dtype=torch.int32)
            path = pydiffvg.Path(num_control_points=num_control_points,
                                 points=points,
                                 stroke_width=stroke_width,
                                 is_closed=True)
        shapes.append(path)

        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(shapes) - 1]),
                                         fill_color=fill_color,
                                         stroke_color=stroke_color)
        shape_groups.append(path_group)
    return canvas_size, canvas_size, shapes, shape_groups


def as_diffvg_render(paths: [dict], background_color: torch.Tensor, canvas_size: int) -> torch.Tensor:
    params = to_diffvg_svg_params(paths, background_color, canvas_size)
    scene_args = pydiffvg.RenderFunction.serialize_scene(*params)
    render = pydiffvg.RenderFunction.apply
    img = render(canvas_size,  # width
                 canvas_size,  # height
                 2,  # num_samples_x
                 2,  # num_samples_y
                 0,  # seed
                 None,  # background_image
                 *scene_args)
    img = img[:, :, :3]  # RGBA -> RGB
    img = img.permute(2, 0, 1)  # HWC -> CHW

    return img


def as_SVGCont(cover: Cover, canvas_size: int):
    image = SVGContainer(width=canvas_size, height=canvas_size)
    backgroundColor = tensor_color_to_int(cover.background_color, 1.0)
    rect = RectTag(attrs_dict={"width": canvas_size, "height": canvas_size,
                               "fill": color_to_rgba_attr(backgroundColor)})
    image.add_inner_node(rect)
    for p in cover.figures:
        points = p.points.round().to(int)  # For `self.path_segment_count_` segments
        stroke_width = p.stroke_width.round().to(int).item()
        stroke_color = tensor_color_to_int(p.stroke_color)
        fill_color = tensor_color_to_int(p.fill_color)

        path = PathTag()

        path.move_to(points[0][0].item(), points[0][1].item())

        segment_count = (len(points) - 1) // 3
        for j in range(segment_count):
            segment_points = points[1 + j * 3: 1 + (j + 1) * 3]
            path.cubic_to(segment_points[0][0].item(),
                          segment_points[0][1].item(),
                          segment_points[1][0].item(),
                          segment_points[1][1].item(),
                          segment_points[2][0].item(),
                          segment_points[2][1].item())
        path.close_path()
        path.add_attrs({"fill": color_to_rgba_attr(fill_color),
                        "stroke": color_to_rgba_attr(stroke_color),
                        "stroke-width": stroke_width,
                        })
        image.add_inner_node(path)
    return image


def as_SVGCont2(paths: [dict], background_color: torch.Tensor, canvas_size: int):
    cover = Cover()
    cover.background_color = background_color
    for p in paths:
        fig = CoverFigure()
        fig.points = p["points"]
        fig.stroke_width = p["stroke_width"]
        fig.stroke_color = p["stroke_color"]
        fig.fill_color = p["fill_color"]
        cover.add_figure(fig)
    return as_SVGCont(cover, canvas_size)
