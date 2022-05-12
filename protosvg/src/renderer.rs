use usvg::SystemFontDB;

pub fn render(svg_xml: &str) -> Result<tiny_skia::Pixmap, String> {
    let mut opt = usvg::Options::default();
    opt.fontdb.load_system_fonts();
    opt.fontdb.set_generic_families();

    let tree = match usvg::Tree::from_str(svg_xml, &opt) {
        Ok(t) => t,
        Err(e) => return Err(e.to_string()),
    };

    let pixmap_size = tree.svg_node().size.to_screen_size();
    let mut pixmap =
        tiny_skia::Pixmap::new(pixmap_size.width(), pixmap_size.height())
            .ok_or("Pixmap creation failed (invalid size?)")?;
    resvg::render(&tree, usvg::FitTo::Original, pixmap.as_mut())
        .ok_or("Failed to render with resvg")?;
    Ok(pixmap)
}
