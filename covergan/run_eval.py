#!/usr/bin/env python3
# coding: utf-8
import numpy as np
import pydiffvg
from PIL import ImageDraw

from captions.models.captioner import Captioner
from fonts_cfg import FONTS
from outer.dataset import read_music_tensor_for_file
from outer.emotions import Emotion, emotion_from_str, emotions_one_hot
from outer.models.my_generator_fixed_six_figs import MyGeneratorFixedSixFigs
from outer.models.my_generator_rand_figure import MyGeneratorRandFigure
from protosvg.client import PSVG
from scripts.bbox_finder.test_bbox_finder import draw_phrases, get_all_boxes_info_to_paste, draw_to_draw_object
from service_utils import *
from utils.bboxes import BBox
from utils.noise import get_noise

# import cairosvg
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPM


def run_track(audio_file_name, track_artist, track_name,
              num_samples=5, output_dir="generated_covers",
              debug=False):
    emotions: [Emotion] = [emotion_from_str(x) for x in ["serious", "wary"]]

    os.makedirs(output_dir, exist_ok=True)
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # gan_weights = "dataset_full_covers/checkpoint/cgan_out.pt"
    gan_weights = "dataset_full_covers/checkpoint/checkpoint_hz.pt"
    gan_weights = "dataset_full_covers/checkpoint/checkpoint_6figs_5depth_512noise.pt"
    gen_type = MyGeneratorRandFigure
    gen_type = MyGeneratorFixedSixFigs
    captioner_weights = "./weights/captioner.pt"
    checkpoint_root_ = "dataset_full_covers/checkpoint/cgan_out_dataset"

    USE_CAPTIONER = False

    font_dir = "./fonts"
    rasterize = True

    z_dim = 512
    disc_slices_ = 6
    audio_embedding_dim = 281 * disc_slices_
    gen_canvas_size_ = 512
    max_stroke_width = 0.01
    captioner_canvas_size_ = 256
    num_captioner_conv_layers = 3
    num_captioner_linear_layers = 2
    generator_ = gen_type(
        z_dim=z_dim,
        audio_embedding_dim=audio_embedding_dim,
        has_emotions=True,
        num_layers=-1,
        canvas_size=gen_canvas_size_,
        path_count=-1,
        path_segment_count=-1,
        max_stroke_width=max_stroke_width
    ).to(device_)
    generator_.eval()

    gan_weights = torch.load(gan_weights, map_location=device_)
    generator_.load_state_dict(gan_weights["0_state_dict"])

    protosvg_address = "localhost:50051"
    psvg_client_ = PSVG(protosvg_address)

    music_tensor = read_music_tensor_for_file(audio_file_name, checkpoint_root_)
    target_count = 24  # 2m = 120s, 120/5
    if len(music_tensor) < target_count:
        music_tensor = music_tensor.repeat(target_count // len(music_tensor) + 1, 1)
    music_tensor = music_tensor[:target_count].float().to(device_)
    music_tensor = music_tensor[:disc_slices_].flatten()
    music_tensor = music_tensor.unsqueeze(dim=0).repeat((num_samples, 1))

    emotions_tensor = emotions_one_hot(emotions).to(device_).unsqueeze(dim=0).repeat((num_samples, 1))
    noise = get_noise(num_samples, z_dim, device=device_)

    diffvg_svg_params_covers, psvg_covers = generator_(noise, music_tensor, emotions_tensor,
                                                       return_psvg=True,
                                                       return_diffvg_svg_params=True,
                                                       use_triad_coloring=False)
    cover_render_pngs = [psvg_client_.render(x) for x in psvg_covers]
    pils_render = [
        png_data_to_pil_image(x, gen_canvas_size_)
        for x in cover_render_pngs
    ]
    if debug:
        for ind, params in enumerate(diffvg_svg_params_covers):
            pydiffvg.save_svg(f'{output_dir}/back_diffvg_svg_{ind}.svg', *params)
        for ind, im in enumerate(psvg_covers):
            res = psvg_client_.convert_to_svg(im)
            with open(f"{output_dir}/back_psvg_svg_{ind}.svg", 'w') as f:
                f.write(res)
        for ind, im in enumerate(pils_render):
            f_name = f"{output_dir}/back_psvg_rasterized_{track_artist}-{track_name}.png"
            im.save(f_name)
            # draw_phrases(track_artist, track_name,
            #              img_path=f_name, output_path=f"{output_dir}/font_{track_artist}-{track_name}.png")

    if USE_CAPTIONER:
        cover_render_pils = [
            png_data_to_pil_image(x, captioner_canvas_size_)
            for x in cover_render_pngs
        ]
        cover_render_tensors = torch.stack([to_tensor(x) for x in cover_render_pils]).to(device_)
        captioner_ = Captioner(
            canvas_size=captioner_canvas_size_,
            num_conv_layers=num_captioner_conv_layers,
            num_linear_layers=num_captioner_linear_layers
        ).to(device_)
        captioner_.eval()
        captioner_weights = torch.load(captioner_weights, map_location=device_)
        captioner_.load_state_dict(captioner_weights["0_state_dict"])

        pos_preds, color_preds = captioner_(cover_render_tensors)
        pos_preds = torch.round(pos_preds * gen_canvas_size_).to(int)
        color_preds = torch.round(color_preds * 255).to(int)
        artist_name_positions, track_name_positions = pos_preds[:, :4], pos_preds[:, 4:]
        artist_name_colors, track_name_colors = color_preds[:, :3], color_preds[:, 3:]

        for i in range(num_samples):
            psvg_cover = psvg_covers[i]
            pil_cover: Image.Image = cover_render_pils[i]
            artist_name_pos: BBox = pos_tensor_to_bbox(artist_name_positions[i])
            track_name_pos: BBox = pos_tensor_to_bbox(track_name_positions[i])
            artist_name_color: Tuple[int, int, int] = tuple(map(int, artist_name_colors[i]))
            track_name_color: Tuple[int, int, int] = tuple(map(int, track_name_colors[i]))

            artist_name_color = ensure_caption_contrast(
                pil_cover,
                artist_name_pos.recanvas(gen_canvas_size_, captioner_canvas_size_),
                artist_name_color
            )
            track_name_color = ensure_caption_contrast(
                pil_cover,
                track_name_pos.recanvas(gen_canvas_size_, captioner_canvas_size_),
                track_name_color
            )

            # Font properties
            artist_font_family = random.choice(FONTS)
            name_font_family = random.choice(FONTS)

            add_caption(
                psvg_cover, font_dir,
                track_artist, artist_name_pos, rgb_tuple_to_int(artist_name_color), artist_font_family,
                track_name, track_name_pos, rgb_tuple_to_int(track_name_color), name_font_family,
                debug=False
            )
    else:
        for i in range(num_samples):
            psvg_cover = psvg_covers[i]
            pil_img = pils_render[i]
            pil_img.save(f"{output_dir}/debug.png")
            draw = ImageDraw.Draw(pil_img, mode='RGB')
            to_draw = get_all_boxes_info_to_paste(track_artist, track_name, np.asarray(pil_img), for_svg=True)
            for x in to_draw:
                draw_to_draw_object(draw, x)
                if x["type"] == "text":
                    l, t = x["text_xy_left_top"]
                    ascent = x["ascent"]
                    t += ascent - 1
                    r, b = x["text_xy_right_bottom"]
                    add_ready_text_node(psvg_cover, x["word"], x["color"], x["font"], (l, b), (r, t))
            pil_img.save(f"{output_dir}/font_{track_artist} - {track_name}.png")

    if rasterize:
        # [(svg_xml: str, png_data: bytes)]
        output_func = psvg_client_.convert_and_render
    else:
        # [svg_xml: str]
        output_func = psvg_client_.convert_to_svg
    result = list(map(output_func, psvg_covers))

    basename = os.path.basename(audio_file_name)
    for i, res in enumerate(result):
        (svg_xml, png_data) = res
        svg_xml = add_font_imports_to_svg_xml(svg_xml)
        svg_cover_filename = f"{output_dir}/psvg_{basename}-{i + 1}.svg"
        with open(svg_cover_filename, 'w', encoding="utf-8") as f:
            f.write(svg_xml)
        if rasterize:
            png_cover_filename = f"{output_dir}/psvg_{basename}-{i + 1}.png"
            with open(png_cover_filename, 'wb') as f:
                f.write(png_data)
            # png_cover_filename_cairo = f"{output_dir}/psvg_cairo_{basename}-{i + 1}.png"
            # drawing = svg2rlg(svg_cover_filename)
            # renderPM.drawToFile(drawing, png_cover_filename_cairo, fmt="PNG")
            # cairosvg.svg2png(url=svg_cover_filename, write_to=png_cover_filename_cairo)


def fname_to_test(fname):
    fname = fname.replace(".pt", "")
    splt = fname.replace(".mp3", "").split(" - ")
    return fname, splt[0], splt[1]


if __name__ == '__main__':
    print("Don't forget to run `protosvg` before!")
    RAND = False
    RAND = True
    tests = [
        # ("&me - The Rapture Pt.II.mp3", "&me", "The Rapture Pt.II"),
        # ("Jason Mraz - I Won't Give Up.mp3", "Jason Mraz", "I Won't Give Up"),
        # ("Afro Nostalgia - Ocean Vibe.mp3", "Afro Nostalgia", "Ocean Vibe"),
        # ("Young & Sick - Sleepyhead.mp3", "Young & Sick", "Sleepyhead"),
        ("Маша и Медведи - Solntseklesh.mp3", "Маша и Медведи", "Solntseklesh"),
        ("Кино - Группа крови.mp3", "Кино", "Группа крови"),
        # ("Will Clarke - Our Love.mp3", "Will Clarke", "Our Love"),
        # ("Peach Face - Ghost.mp3", "Peach Face", "Ghost"),
        ("KYUL - Even Though We Are Not the Same 우리의 감정이 같을 순 없지만.mp3",
         "KYUL", "Even Though We Are Not the Same 우리의 감정이 같을 순 없지만"),
    ]
    if RAND:
        all_fnames = os.listdir("dataset_full_covers/checkpoint/cgan_out_dataset")
        tests = [
            fname_to_test(random.choice(all_fnames))
            for _ in range(10)
        ]
    for (audio_file_name, track_artist, track_name) in tests:
        run_track(audio_file_name, track_artist, track_name,
                  num_samples=1,
                  output_dir="generated_covers_tests")
