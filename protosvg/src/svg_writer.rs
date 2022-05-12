use std::fmt;
use xmlwriter::*;
use htmlescape::encode_minimal;

use crate::protos::protosvg::*;

impl fmt::Display for Point {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{},{}", self.x, self.y)
    }
}

impl fmt::Display for font::FontStyle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use font::FontStyle::*;
        write!(
            f,
            "{}",
            match *self {
                NormalStyle => "normal",
                Italic => "italic",
                Oblique => "oblique",
            }
        )
    }
}

impl fmt::Display for font::FontStretch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use font::FontStretch::*;
        write!(
            f,
            "{}",
            match *self {
                NormalStretch => "normal",
                UltraCondensed => "ultra-condensed",
                ExtraCondensed => "extra-condensed",
                Condensed => "condensed",
                SemiCondensed => "semi-condensed",
                SemiExpanded => "semi-expanded",
                Expanded => "expanded",
                ExtraExpanded => "extra-expanded",
                UltraExpanded => "ultra-expanded",
            }
        )
    }
}

impl fmt::Display for label::LengthAdjust {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use label::LengthAdjust::*;
        write!(
            f,
            "{}",
            match *self {
                Spacing => "spacing",
                SpacingAndGlyphs => "spacingAndGlyphs",
            }
        )
    }
}

impl fmt::Display for label::WritingMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use label::WritingMode::*;
        write!(
            f,
            "{}",
            // SVG 2 -> SVG 1.1
            // See https://www.w3.org/TR/SVG2/text.html#WritingModeProperty
            match *self {
                HorizontalTopBottom => "lr",
                VerticalRightLeft => "tb",
                VerticalLeftRight => "tb",
            }
        )
    }
}

impl fmt::Display for base_gradient::SpreadMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use base_gradient::SpreadMethod::*;
        write!(
            f,
            "{}",
            match *self {
                Pad => "pad",
                Reflect => "reflect",
                Repeat => "repeat",
            }
        )
    }
}

impl fmt::Display for stroke::StrokeLineJoin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use stroke::StrokeLineJoin::*;
        write!(
            f,
            "{}",
            match *self {
                Miter => "miter",
                Round => "round",
                Bevel => "bevel",
            }
        )
    }
}

fn write_point_attr(
    point: &Point,
    x_attr: &str,
    y_attr: &str,
    w: &mut XmlWriter,
) {
    w.write_attribute(x_attr, &point.x);
    w.write_attribute(y_attr, &point.y);
}

fn to_svg_color(color: &Color) -> (String, f32) {
    let rgba = color.rgba;
    let (r, g, b, a) = match rgba.to_le_bytes()[..] {
        [r, g, b, a] => (r, g, b, a),
        _ => unreachable!(),
    };
    (format!("#{:02X}{:02X}{:02X}", r, g, b), a as f32 / 255.0)
}

fn to_font_style(font: &Font) -> Result<String, String> {
    let mut result = Vec::new();

    let size = font.size;
    if size != 0.0 {
        if size < 0.0 {
            return Err("Negative font size".to_owned());
        }
        result.push(format!("font-size:{}px", size));
    }

    let weight = font.weight;
    if weight != 0 {
        if (weight % 100 != 0) || (weight / 100 < 1) || (weight / 100 > 9) {
            return Err("Font weight is not one of {100, …, 900}".to_owned());
        }
        result.push(format!("font-weight:{}", weight));
    }

    let family = &font.family;
    if !family.is_empty() {
        result.push(format!("font-family:{}", family));
    }

    result.push(format!("font-style:{}", font.style()));
    result.push(format!("font-stretch:{}", font.stretch()));

    if font.small_caps {
        result.push("font-variant:small-caps".to_owned());
    }

    Ok(result.join(";"))
}

fn to_transform_string(transform: &Transform) -> String {
    format!(
        "matrix({},{},{},{},{},{})",
        transform.a,
        transform.b,
        transform.c,
        transform.d,
        transform.e,
        transform.f,
    )
}

trait SvgWritable {
    // General ideas for the implementation:
    // * Elements are started and ended by calling functions, not by `write_svg`
    // * However, `write_svg` might start children elements, meaning that it
    //   needs to be called after any attributes of the element are written
    //   by the calling function.
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String>;
}

impl SvgWritable for Path {
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String> {
        let mut segment_strings = vec![];
        for segment in &self.segments {
            use path_segment::Segment::*;
            let (cmd, data) = match &segment.segment {
                Some(Arc(arc)) => (
                    "A",
                    format!(
                        "{} {} {} {} {} {}",
                        arc.radius_x,
                        arc.radius_y,
                        arc.x_axis_rotation,
                        arc.large_arc as u32,
                        arc.sweep as u32,
                        arc.end.as_ref().ok_or("Arc segment to nowhere")?
                    ),
                ),
                Some(Cubic(cubic)) => (
                    "C",
                    format!(
                        "{} {} {}",
                        cubic
                            .start_control
                            .as_ref()
                            .ok_or("Cubic segment without startControl")?,
                        cubic
                            .end_control
                            .as_ref()
                            .ok_or("Cubic segment without endControl")?,
                        cubic
                            .end
                            .as_ref()
                            .ok_or("Cubic segment without end")?
                    ),
                ),
                Some(Move(mv)) => (
                    "M",
                    mv.to
                        .as_ref()
                        .ok_or("Move segment to nowhere")?
                        .to_string(),
                ),
                Some(Line(line)) => (
                    "L",
                    line.to
                        .as_ref()
                        .ok_or("Line segment to nowhere")?
                        .to_string(),
                ),
                Some(Quad(quad)) => (
                    "Q",
                    format!(
                        "{} {}",
                        quad.control
                            .as_ref()
                            .ok_or("Quad segment without control")?,
                        quad.end.as_ref().ok_or("Quad segment without end")?
                    ),
                ),
                Some(Close(_)) => ("Z", String::new()),
                None => {
                    return Err("Invalid unspecified PathSegment".to_owned())
                }
            };
            let cmd = if segment.relative {
                cmd.to_lowercase()
            } else {
                cmd.to_owned()
            };
            segment_strings.push(format!("{} {}", cmd, data));
        }
        w.write_attribute("d", &segment_strings.join("\n"));

        Ok(())
    }
}

impl SvgWritable for Rectangle {
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String> {
        write_point_attr(
            self.start
                .as_ref()
                .ok_or("Rectangle has unspecified start coordinate")?,
            "x",
            "y",
            w,
        );

        w.write_attribute("width", &self.width);
        w.write_attribute("height", &self.height);
        w.write_attribute("rx", &self.corner_radius_x);
        w.write_attribute("ry", &self.corner_radius_y);

        Ok(())
    }
}

impl SvgWritable for Ellipse {
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String> {
        write_point_attr(
            self.center
                .as_ref()
                .ok_or("Ellipse has unspecified center coordinate")?,
            "cx",
            "cy",
            w,
        );

        w.write_attribute("rx", &self.radius_x);
        w.write_attribute("ry", &self.radius_y);

        Ok(())
    }
}

impl SvgWritable for Square {
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String> {
        write_point_attr(
            self.start
                .as_ref()
                .ok_or("Square has unspecified start coordinate")?,
            "x",
            "y",
            w,
        );

        w.write_attribute("width", &self.width);
        w.write_attribute("height", &self.width);
        w.write_attribute("rx", &self.corner_radius_x);
        w.write_attribute("ry", &self.corner_radius_y);

        Ok(())
    }
}

impl SvgWritable for Circle {
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String> {
        write_point_attr(
            self.center
                .as_ref()
                .ok_or("Circle has unspecified center coordinate")?,
            "cx",
            "cy",
            w,
        );

        w.write_attribute("r", &self.radius);

        Ok(())
    }
}

impl SvgWritable for Label {
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String> {
        let start = self.start
                        .as_ref()
                        .ok_or("Label has unspecified start coordinate")?;
        write_point_attr(start, "x", "y", w);

        if let Some(font) = &self.font {
            let font_style = to_font_style(font)?;
            w.write_attribute("style", &font_style);
        }
        if self.character_rotation != 0 {
            w.write_attribute("rotate", &self.character_rotation);
        }
        if self.text_length != 0 {
            w.write_attribute("textLength", &self.text_length);
        }

        w.write_attribute("lengthAdjust", &self.length_adjust());
        w.write_attribute("writing-mode", &self.writing_mode());

        let text = encode_minimal(&self.text);
        if text.contains("\n") {
            let mut spacing = "0";
            for line in text.split("\n") {
                w.start_element("tspan");
                use label::WritingMode::HorizontalTopBottom;
                if self.writing_mode() == HorizontalTopBottom {
                    w.write_attribute("x", &start.x);
                    w.write_attribute("dy", &spacing);
                    spacing = "1.2em";
                } else {
                    w.write_attribute("y", &start.y);
                    w.write_attribute("dx", &spacing);
                    spacing = "-1.2em";
                }
                w.write_text(line);
                w.end_element();
            }
        } else {
            w.write_text(&text);
        }

        Ok(())
    }
}

impl SvgWritable for GradientStop {
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String> {
        let offset = self.offset;
        if !(0.0..=1.0).contains(&offset) {
            return Err("Gradient offset is not in range [0; 1]".to_owned());
        }
        w.write_attribute("offset", &offset);

        if let Some(color) = &self.color {
            let (stop_color, stop_opacity) = to_svg_color(color);
            w.write_attribute("stop-color", &stop_color);
            w.write_attribute("stop-opacity", &stop_opacity);
        }

        Ok(())
    }
}

impl SvgWritable for BaseGradient {
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String> {
        w.write_attribute("gradientUnits", "userSpaceOnUse");

        if let Some(transform) = &self.transform {
            w.write_attribute(
                "gradientTransform",
                &to_transform_string(transform),
            );
        }

        w.write_attribute("spreadMethod", &self.spread_method());

        for stop in &self.stops {
            w.start_element("stop");
            stop.write_svg(w)?;
            w.end_element();
        }

        Ok(())
    }
}

impl SvgWritable for LinearGradient {
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String> {
        if let Some(start) = &self.start {
            write_point_attr(start, "x1", "y1", w);
        }
        if let Some(end) = &self.end {
            write_point_attr(end, "x2", "y2", w);
        }

        self.base
            .as_ref()
            .ok_or("Invalid gradient without base")?
            .write_svg(w)?;

        Ok(())
    }
}

impl SvgWritable for RadialGradient {
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String> {
        if let Some(start) = &self.start {
            write_point_attr(start, "fx", "fy", w);
        }
        if let Some(end) = &self.end {
            write_point_attr(end, "cx", "cy", w);
        }

        if self.start_radius != 0 {
            w.write_attribute("fr", &self.start_radius);
        }
        if self.end_radius != 0 {
            w.write_attribute("r", &self.end_radius);
        }

        self.base
            .as_ref()
            .ok_or("Invalid gradient without base")?
            .write_svg(w)?;

        Ok(())
    }
}

impl SvgWritable for Stroke {
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String> {
        let (str_color, str_opacity) =
            to_svg_color(self.color.as_ref().ok_or("Stroke without a color")?);
        w.write_attribute("stroke", &str_color);
        w.write_attribute("stroke-opacity", &str_opacity);

        w.write_attribute("stroke-width", &self.width);
        w.write_attribute("stroke-linejoin", &self.line_join());

        Ok(())
    }
}

impl SvgWritable for Shape {
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String> {
        if let Some(stroke) = &self.stroke {
            stroke.write_svg(w)?;
        }
        if let Some(transform) = &self.transform {
            w.write_attribute("transform", &to_transform_string(transform));
        }

        use shape::Shape::*;
        match &self.shape {
            Some(Circle(c)) => c.write_svg(w),
            Some(Label(l)) => l.write_svg(w),
            Some(Ellipse(e)) => e.write_svg(w),
            Some(Path(p)) => p.write_svg(w),
            Some(Rectangle(r)) => r.write_svg(w),
            Some(Square(s)) => s.write_svg(w),
            None => return Err("Invalid Shape without shape".to_owned()),
        }?;

        Ok(())
    }
}

impl SvgWritable for ProtoSvg {
    fn write_svg(&self, w: &mut XmlWriter) -> Result<(), String> {
        // Namespace
        w.write_attribute("xmlns", "http://www.w3.org/2000/svg");

        // Canvas
        let canvas_width = self.width;
        let canvas_height = self.height;
        w.write_attribute("width", &canvas_width);
        w.write_attribute("height", &canvas_height);
        w.write_attribute_fmt(
            "viewBox",
            format_args!("{} {} {} {}", 0, 0, canvas_width, canvas_height),
        );

        // Write all used gradient to the <defs> section.
        // Duplicates are possible and are not handled.
        let mut last_used_id = 0;
        w.start_element("defs");
        for shape in &self.shapes {
            use shape::Fill::*;
            match &shape.fill {
                Some(LinearGradient(lg)) => {
                    w.start_element("linearGradient");
                    w.write_attribute_fmt(
                        "id",
                        format_args!("id{}", last_used_id),
                    );
                    last_used_id += 1;
                    lg.write_svg(w)?;
                    w.end_element()
                }
                Some(RadialGradient(rg)) => {
                    w.start_element("radialGradient");
                    w.write_attribute_fmt(
                        "id",
                        format_args!("id{}", last_used_id),
                    );
                    last_used_id += 1;
                    rg.write_svg(w)?;
                    w.end_element()
                }
                _ => {}
            };
        }
        w.end_element();

        // Write the convenience background rectangle if the color is set.
        if let Some(background_color) = &self.background_color {
            w.start_element("rect");
            w.write_attribute("width", &canvas_width);
            w.write_attribute("height", &canvas_height);
            let (fill, opacity) = to_svg_color(background_color);
            w.write_attribute("fill", &fill);
            w.write_attribute("fill-opacity", &opacity);
            w.end_element();
        }

        // Write the actual shapes.
        last_used_id = 0;
        for shape in &self.shapes {
            use shape::Shape::*;
            let shape_node_name = match &shape.shape {
                Some(Circle(_)) => "circle",
                Some(Label(_)) => "text",
                Some(Ellipse(_)) => "ellipse",
                Some(Path(_)) => "path",
                Some(Rectangle(_)) => "rect",
                Some(Square(_)) => "rect",
                None => return Err("Invalid Shape without shape".to_owned()),
            };
            w.start_element(&shape_node_name);

            // Here we reference existing ids from the <defs> section, as the
            // traversal order is unchanged.
            use shape::Fill::*;
            match &shape.fill {
                Some(LinearGradient(_)) | Some(RadialGradient(_)) => {
                    w.write_attribute_fmt(
                        "fill",
                        format_args!("url(#id{})", last_used_id),
                    );
                    last_used_id += 1;
                }
                Some(Color(c)) => {
                    let (fill, opacity) = to_svg_color(c);
                    w.write_attribute("fill", &fill);
                    w.write_attribute("fill-opacity", &opacity);
                }
                None => {}
            };

            shape.write_svg(w)?;
            w.end_element();
        }

        Ok(())
    }
}

pub fn protosvg_to_svg(svg: ProtoSvg) -> Result<String, String> {
    let opt = Options::default();
    let mut w = XmlWriter::new(opt);

    w.start_element("svg");
    svg.write_svg(&mut w)?;
    w.end_element();

    Ok(w.end_document())
}
