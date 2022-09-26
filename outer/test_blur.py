def blr():
    from PIL import Image
    from torchvision.transforms import transforms
    from torchvision.transforms.functional import gaussian_blur

    im = Image.open("../dataset_emoji_4/clean_covers/demo_track_1.jpg")
    convert_tensor = transforms.ToTensor()
    tens = convert_tensor(im)
    blur = gaussian_blur(tens, kernel_size=29)
    to_pil = transforms.ToPILImage()
    pil = to_pil(blur)
    pil.save('out2.png')


def wand_rend_test():
    from outer.SVGContainer import wand_rendering
    f_name = "../generated_covers_tests_rand2/Sound Quelle - Volga-1.svg"
    f_name = "kek3.svg"
    with open(f_name, mode="r", encoding="utf-8") as f:
        img = wand_rendering(f.read())
    with open("test.png", mode="wb") as f:
        f.write(img)


def svglib_rend_test():
    from outer.SVGContainer import svglib_rendering, svglib_rendering_from_file
    # f_name = "../generated_covers_svgcont2/&me - The Rapture &&%$Pt.II-1.svg"
    f_name = "kek2.svg"
    with open(f_name, mode="r", encoding="utf-8") as f:
        img = svglib_rendering_from_file(f.name)
    with open("test.png", mode="wb") as f:
        f.write(img)


def load_test():
    from outer.SVGContainer import SVGContainer
    data = "mega_kek.svg"
    svg_cont = SVGContainer.load_svg(open(data).read())
    svg_cont.save_png("kek3.png", renderer_type="wand")
    svg_cont.save_svg("kek3_.svg")


def add_text_test():
    from outer.SVGContainer import SVGContainer
    from service_utils import paste_caption
    data = "kek2_notext.svg"
    svg_cont = SVGContainer.load_svg(open(data).read())
    pil = svg_cont.to_PIL(renderer_type="wand").convert("RGB")
    paste_caption(svg_cont, pil, "Nyash", "Myash", "../fonts")
    pil.save("kek2_with_text.png")
    # svg_cont.save_png("kek2_with_text.png", renderer_type="wand")
    svg_cont.save_svg("kek2_with_text.svg")


def check_to_obj():
    from outer.SVGContainer import SVGContainer
    data = "mega_kek.svg"
    svg_cont = SVGContainer.load_svg(open(data).read())
    obj = svg_cont.to_obj()
    import json
    res = json.dumps(obj, indent=4)
    print(res)


# wand_rend_test()
# load_test()
# add_text_test()
check_to_obj()
