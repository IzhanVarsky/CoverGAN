import html
from typing import List


def wand_rendering(svg_str):
    # WARNING! Works only on Windows.
    import wand.image
    with wand.image.Image(blob=svg_str.encode(), format="svg") as image:
        png_image = image.make_blob("png")
    return png_image


def cairo_rendering(svg_str):
    # WARNING! Works only on Linux.
    from cairosvg import svg2png
    return svg2png(bytestring=svg_str)


def svglib_rendering(svg_str):
    # WARNING! <style> tag is not supported!
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(delete=False) as f:
        tmp_filename = f.name
        f.write(svg_str.encode("utf-8"))
    res = svglib_rendering_from_file(tmp_filename)
    os.remove(tmp_filename)
    return res


def svglib_rendering_from_file(svg_filename):
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPM
    drawing = svg2rlg(svg_filename)
    return renderPM.drawToString(drawing, fmt="PNG")


class SVGNode:
    def __str__(self):
        raise Exception("__str__() not implemented")

    def to_obj(self):
        raise Exception("to_obj() not implemented")


class TextNode(SVGNode):
    def __init__(self, text=""):
        self.text = text

    def replace_text(self, new_text):
        self.text = new_text

    def escape_text(self):
        return html.escape(self.text)

    def __str__(self):
        return self.escape_text()

    def to_obj(self):
        return {"#text": str(self)}


class BasicSVGTag(SVGNode):
    def __init__(self, tag_name: str, attrs_dict=None):
        super().__init__()
        self.tag_name = tag_name
        self.attrs: dict = attrs_dict if attrs_dict is not None else {}

    def add_attr(self, attr_name, attr_value):
        self.attrs[attr_name] = attr_value

    def add_attrs(self, d: dict):
        for p in d:
            self.add_attr(p, d[p])

    def tag_with_attrs(self):
        attrs = " ".join(list(map(lambda p: f'{p}="{self.attrs[p]}"', self.attrs))).strip()
        if len(attrs) > 0:
            attrs = " " + attrs
        return f"{self.tag_name}{attrs}"


class OpenSVGTag(BasicSVGTag):
    def __init__(self, tag_name: str, attrs_dict=None):
        super().__init__(tag_name=tag_name, attrs_dict=attrs_dict)
        self.inner_nodes: List[SVGNode] = []

    def insert_inner_node(self, index, node: SVGNode):
        self.inner_nodes.insert(index, node)

    def add_inner_node(self, node: SVGNode):
        self.inner_nodes.append(node)

    def inner_nodes_to_string(self):
        # return "\n".join(list(map(str, self.inner_nodes))).strip()
        return "".join(list(map(str, self.inner_nodes)))

    def __str__(self):
        return f"<{self.tag_with_attrs()}>{self.inner_nodes_to_string()}</{self.tag_name}>"

    def to_obj(self):
        return {self.tag_name: {"attrs": [self.attrs],
                                "children": [node.to_obj() for node in self.inner_nodes]}}


class CloseSVGTag(BasicSVGTag):
    def __init__(self, tag_name: str, attrs_dict=None):
        super().__init__(tag_name=tag_name, attrs_dict=attrs_dict)

    def __str__(self):
        return f"<{self.tag_with_attrs()}/>"

    def to_obj(self):
        return {self.tag_name: {"attrs": [self.attrs]}}


class RawTextTag(OpenSVGTag):
    def __init__(self, tag_name, text: str, attrs_dict=None):
        super().__init__(tag_name=tag_name, attrs_dict=attrs_dict)
        self.add_inner_node(TextNode(text))


class TextTag(RawTextTag):
    def __init__(self, text: str = "", attrs_dict=None):
        super().__init__(tag_name="text", text=text, attrs_dict=attrs_dict)


class StyleTag(RawTextTag):
    def __init__(self, text: str = "", attrs_dict=None):
        super().__init__(tag_name="style", text=text, attrs_dict=attrs_dict)


class FontImporter(StyleTag):
    def __init__(self, fonts=None):
        super().__init__(text="")
        self.fonts = set(fonts if fonts is not None else [])

    def add_font(self, font_style: str):
        self.fonts.add(font_style)

    def get_text(self):
        return "\n".join(list(map(lambda f: f"@import url('https://fonts.googleapis.com/css?family={f}');",
                                  self.fonts)))

    def __str__(self):
        text = self.get_text()
        if text == "":
            return ""
        node = TextNode(text)
        return f"<{self.tag_with_attrs()}>{node}</{self.tag_name}>"

    def to_obj(self):
        text = self.get_text()
        if text == "":
            return None
        node = TextNode(text)
        return {self.tag_name: {"attrs": [self.attrs],
                                "children": [node.to_obj()]}}


class RectTag(CloseSVGTag):
    def __init__(self, attrs_dict=None):
        super().__init__(tag_name="rect", attrs_dict=attrs_dict)


class CircleTag(CloseSVGTag):
    def __init__(self, attrs_dict=None):
        super().__init__(tag_name="circle", attrs_dict=attrs_dict)

    @classmethod
    def create(cls, r, cx, cy, color):
        return CircleTag({"r": r, "cx": cx, "cy": cy, "fill": color})


class PathTag(CloseSVGTag):
    def __init__(self, attrs_dict=None):
        super().__init__(tag_name="path", attrs_dict=attrs_dict)
        if "d" not in self.attrs:
            self.attrs["d"] = ""

    def prepend(self):
        if self.attrs["d"] == "":
            prepend = ""
        else:
            prepend = " "
        return prepend

    def move_to(self, x, y):
        self.attrs["d"] += f"{self.prepend()}M {x},{y}"

    def cubic_to(self, startControlX, startControlY, endControlX, endControlY, endX, endY):
        self.attrs["d"] += \
            f"{self.prepend()}C {startControlX},{startControlY} {endControlX},{endControlY} {endX},{endY}"

    def close_path(self):
        self.attrs["d"] += f"{self.prepend()}Z"


class RadialGradientTag(OpenSVGTag):
    def __init__(self, attrs_dict=None):
        super().__init__(tag_name="radialGradient", attrs_dict=attrs_dict)

    def add_stop(self, offset, stop_color):
        t = CloseSVGTag("stop", {"offset": offset, "stop-color": stop_color})
        self.add_inner_node(t)


class SVGContainer(OpenSVGTag):
    def __init__(self, width, height, use_font_importer=True):
        self.width = width
        self.height = height
        attrs_dict = {"xmlns": "http://www.w3.org/2000/svg",
                      "width": width,
                      "height": height,
                      "viewBox": f"0 0 {width} {height}",
                      }
        super().__init__(tag_name="svg", attrs_dict=attrs_dict)
        if use_font_importer:
            self.font_importer = FontImporter()
            self.add_inner_node(self.font_importer)
        self.free_id_num = 0

    def bind_tags_with_id(self, tag1: BasicSVGTag, tag2: BasicSVGTag, field_name):
        tag_id = f"id{self.free_id_num}"
        tag1.add_attr(field_name, f"url(#{tag_id})")
        tag2.add_attr("id", tag_id)
        self.free_id_num += 1

    def save_svg(self, out_filename: str):
        with open(out_filename, mode="w", encoding="utf-8") as f:
            f.write(str(self))

    def render(self, renderer_type):
        if renderer_type in mp:
            return mp[renderer_type](str(self))
        raise Exception(f"Unexpected renderer_type `{renderer_type}`.")

    def to_PIL(self, renderer_type):
        png = self.render(renderer_type)
        from PIL import Image
        import io
        return Image.open(io.BytesIO(png))

    def save_png(self, out_filename: str, renderer_type: str):
        with open(out_filename, "wb") as out:
            out.write(self.render(renderer_type))

    @classmethod
    def load_svg(cls, svg_str, use_font_importer=False):
        from xml.dom import minidom
        tree = minidom.parseString(svg_str)
        root = tree.getElementsByTagName("svg")[0]
        root_attrs = dict(root.attributes.items())
        res = SVGContainer(root_attrs["width"], root_attrs["height"], use_font_importer=use_font_importer)
        res.add_attrs(root_attrs)

        def traverse_tree(parent: OpenSVGTag, node):
            for x in node.childNodes:
                continue_loop = False
                if x.nodeName == "#text":
                    child = TextNode(x.nodeValue)
                elif len(x.childNodes) == 0:
                    child = CloseSVGTag(x.nodeName)
                else:
                    child = OpenSVGTag(x.nodeName)
                    continue_loop = True
                if x.attributes is not None:
                    child.add_attrs(dict(x.attributes.items()))
                parent.add_inner_node(child)
                if continue_loop:
                    traverse_tree(child, x)

        traverse_tree(res, root)
        return res


mp = {"wand": wand_rendering, "cairo": cairo_rendering}
