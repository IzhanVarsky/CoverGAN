#!/usr/bin/env python3
# coding: utf-8

from captions.models.captioner import Captioner
from colorer.test_model import get_palette_predictor
from outer.dataset import read_music_tensor_for_file
from outer.emotions import Emotion, emotion_from_str, emotions_one_hot
from outer.models.my_gen_fixed_6figs32_good import MyGeneratorFixedSixFigs32Good
from outer.models.my_generator_fixed_six_figs import MyGeneratorFixedSixFigs
from outer.models.my_generator_fixed_six_figs_backup import MyGeneratorFixedSixFigs32
from outer.models.my_generator_rand_figure import MyGeneratorRandFigure
from outer.models.my_generator_three_figs import MyGeneratorFixedThreeFigs32
from service_utils import *
from utils.bboxes import BBox
from utils.noise import get_noise


def run_track(audio_file_name, track_artist, track_name, emotions=None,
              num_samples=5, output_dir="generated_covers",
              debug=False, num_start=0, gen_canvas_size_=512, deterministic=False,
              apply_filters=False):
    if emotions is None:
        emotions = ["joy"]
    emotions: [Emotion] = [emotion_from_str(x) for x in emotions]

    os.makedirs(output_dir, exist_ok=True)
    device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # gan_weights = "dataset_full_covers/checkpoint/cgan_out.pt"
    a = ("dataset_full_covers/checkpoint/checkpoint_hz.pt",
         MyGeneratorRandFigure,
         512)
    b = ("dataset_full_covers/checkpoint/checkpoint_6figs_5depth_512noise.pt",
         MyGeneratorFixedSixFigs,
         512)
    c = ("dataset_full_covers/checkpoint/cgan_out-100.pt",
         MyGeneratorFixedSixFigs32,
         32)
    d = ("dataset_full_covers/checkpoint/cgan_out-1200.pt",
         MyGeneratorFixedSixFigs32,
         32)
    e = ("dataset_full_covers/checkpoint/cgan_out-640.pt",
         MyGeneratorFixedThreeFigs32,
         32)
    f = ("dataset_full_covers/checkpoint/cgan_6figs_32noise_separated_palette_tanh_betas-880.pt",
         MyGeneratorFixedSixFigs32Good,
         32)
    gan_weights, gen_type, z_dim = a
    captioner_weights = "./weights/captioner.pt"
    checkpoint_root_ = "dataset_full_covers/checkpoint/cgan_out_dataset"
    # checkpoint_root_ = "diploma_test/test_music_ckpts"
    # checkpoint_root_ = "diploma_test/test_random_music/music_ckpts"
    # checkpoint_root_ = "diploma_test/speed_musics_ckpts"
    # checkpoint_root_ = "diploma_test/speed_musics_ckpts_TS"
    # checkpoint_root_ = "diploma_test/vol_musics_ckpts"
    # checkpoint_root_ = "diploma_test/vol_music_ckpts_kino"
    # checkpoint_root_ = "diploma_test/vol_musics_LP_ckpts"

    USE_CAPTIONER = False

    font_dir = "./fonts"
    font_dir = r"F:\stazhirovka2021\Diploma\all_ttfs"
    rasterize = False

    disc_slices_ = 6
    audio_embedding_dim = 281 * disc_slices_
    max_stroke_width = 0.01
    captioner_canvas_size_ = 256
    num_captioner_conv_layers = 3
    num_captioner_linear_layers = 2
    generator_ = gen_type(
        z_dim=z_dim,
        audio_embedding_dim=audio_embedding_dim,
        has_emotions=True,
        num_layers=5,
        canvas_size=gen_canvas_size_,
        path_count=-1,
        path_segment_count=-1,
        max_stroke_width=max_stroke_width
    ).to(device_)
    generator_.eval()

    gan_weights = torch.load(gan_weights, map_location=device_)
    generator_.load_state_dict(gan_weights["0_state_dict"])

    USE_PALETTE = True
    USE_TRIAD = False
    palette_generator = get_palette_predictor(device_, model_type="1")

    def generate(z, audio_embedding_disc, emotions, **kwargs):
        if not USE_PALETTE or palette_generator is None:
            return generator_(z, audio_embedding_disc, emotions, **kwargs)
        return generator_(z, audio_embedding_disc, emotions, palette_generator=palette_generator, **kwargs)

    music_tensor = read_music_tensor_for_file(audio_file_name, checkpoint_root_)
    target_count = 24  # 2m = 120s, 120/5
    if len(music_tensor) < target_count:
        music_tensor = music_tensor.repeat(target_count // len(music_tensor) + 1, 1)
    music_tensor = music_tensor[:target_count].float().to(device_)
    music_tensor = music_tensor[:disc_slices_].flatten()
    music_tensor = music_tensor.unsqueeze(dim=0).repeat((num_samples, 1))

    emotions_tensor = emotions_one_hot(emotions).to(device_).unsqueeze(dim=0).repeat((num_samples, 1))
    noise = get_noise(num_samples, z_dim, device=device_)
    if deterministic:
        noise = noise * 0

    psvg_covers: [SVGContainer] = generate(noise, music_tensor, emotions_tensor,
                                           return_psvg=True,
                                           use_triad_coloring=USE_TRIAD)

    if apply_filters:
        filtered_samples = round(num_samples // 2)
        filters = apply_filters if isinstance(apply_filters, list) else list(OverlayFilter)
        for psvg_cover in psvg_covers[-filtered_samples:]:
            overlay_filter = random.choice(filters)
            add_filter(psvg_cover, overlay_filter)

    # diffvg_svg_params_covers, psvg_covers = generate(noise, music_tensor, emotions_tensor,
    #                                                  return_psvg=True,
    #                                                  return_diffvg_svg_params=True,
    #                                                  use_triad_coloring=USE_TRIAD)
    # cover_render_pngs = [psvg_client_.render(x) for x in psvg_covers]
    cover_render_pngs = [x.render(renderer_type='wand') for x in psvg_covers]
    pils_render = [
        png_data_to_pil_image(x, gen_canvas_size_)
        for x in cover_render_pngs
    ]
    if debug:
        # for ind, params in enumerate(diffvg_svg_params_covers):
        #     pydiffvg.save_svg(f'{output_dir}/back_diffvg_svg_{ind}.svg', *params)
        # for ind, im in enumerate(psvg_covers):
        #     res = psvg_client_.convert_to_svg(im)
        #     with open(f"{output_dir}/back_psvg_svg_{ind}.svg", 'w') as f:
        #         f.write(res)
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

            add_caption(
                psvg_cover, font_dir,
                track_artist, artist_name_pos, artist_name_color,
                track_name, track_name_pos, track_name_color,
                debug=False
            )
    else:
        for i in range(num_samples):
            psvg_cover = psvg_covers[i]
            pil_img = pils_render[i]
            pil_img.save(f"{output_dir}/debug.png")
            track_name2 = track_name.split(" x")[0]
            track_name2 = track_name2.split(" vol")[0]
            paste_caption(psvg_cover, pil_img, track_artist, track_name2, font_dir, for_svg=True)
            # pil_img.save(f"{output_dir}/font_{track_artist} - {track_name}.png")
            pil_img.save(f"{output_dir}/{track_artist} - {track_name}-{num_start + i + 1}.png")

    if rasterize:
        # [(svg_xml: str, png_data: bytes)]
        output_func = lambda x: (str(x), x.render())
    else:
        # [svg_xml: str]
        output_func = str
    result = list(map(output_func, psvg_covers))

    basename = os.path.basename(audio_file_name)
    for i, res in enumerate(result):
        i += num_start
        if rasterize:
            (svg_xml, png_data) = res
        else:
            svg_xml = res
        # svg_cover_filename = f"{output_dir}/psvg_{basename}-{i + 1}.svg"
        svg_cover_filename = f"{output_dir}/{track_artist} - {track_name}-{i + 1}.svg"
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


def test1():
    RAND = False
    # RAND = True
    tests = [
        ("&me - The Rapture Pt.II.mp3", "&me", "The Rapture &&%$Pt.II"),
        ("Jason Mraz - I Won't Give Up.mp3", "Jason Mraz", "I Won't Give Up"),
        ("Afro Nostalgia - Ocean Vibe.mp3", "Afro Nostalgia", "Ocean Vibe"),
        ("Young & Sick - Sleepyhead.mp3", "Young & Sick", "Sleepyhead"),
        ("Маша и Медведи - Solntseklesh.mp3", "Маша и Медведи", "Solntseklesh"),
        ("Кино - Группа крови.mp3", "Кино", "Группа крови"),
        ("Will Clarke - Our Love.mp3", "Will Clarke", "Our Love"),
        ("Peach Face - Ghost.mp3", "Peach Face", "Ghost"),
        ("KYUL - Even Though We Are Not the Same 우리의 감정이 같을 순 없지만.mp3",
         "KYUL", "Even Though We Are Not the Same 우리의 감정이 같을 순 없지만"),
    ]
    if RAND:
        all_fnames = os.listdir("dataset_full_covers/checkpoint/cgan_out_dataset")
        tests = [
            fname_to_test(random.choice(all_fnames))
            for _ in range(15)
        ]
    for (audio_file_name, track_artist, track_name) in tests:
        run_track(audio_file_name, track_artist, track_name,
                  num_samples=6,
                  output_dir="generated_covers_svgcont9")


def test2():
    tests = [
        ("24kGoldn_-_Love_Or_Lust_72874141.mp3", "24kGoldn", "Love Or Lust"),
        ("Burak_Yeter_-_Body_Talks_71511893.mp3", "Burak Yeter", "Body Talks"),
        ("Dotan_-_Numb_68724683.mp3", "Dotan", "Numb"),
        ("Ed_Sheeran_-_Afterglow_72055475.mp3", "Ed Sheeran", "Afterglow"),
        ("Foushee_-_Deep_End_70581187.mp3", "Foushee", "Deep End"),
        ("Helena_Arash_-_Angels_Lullaby_72911174.mp3", "Arash", "Angels Lullaby"),
        ("Imagine_Dragons_-_Follow_You_72848541 (1).mp3", "Imagine Dragons", "Follow You"),
        ("Minelli_-_Rampampam_72874060.mp3", "Minelli", "Rampampam"),
        ("Pnk_Willow_Sage_Hart_-_Cover_Me_In_Sunshine_72663179.mp3", "P!nk", "Cover Me In Sunshine"),
        ("Rita_Ora_Imanbek_-_Bang_Bang_72665055.mp3", "Rita Ora", "Bang Bang"),
        ("Riton_x_Nightcrawlers_-_Friday_feat_Mufasa_Hypeman_Dopamine_Re-edit_73265875.mp3",
         "Nightcrawlers", "Friday"),
        ("Nessa_Barrett_jxdn_-_la_di_die_72757984.mp3", "Nessa Barrett", "la di die"),
        ("Tesher_-_Jalebi_Baby_72865842.mp3", "Tesher", "Jalebi Baby"),
        ("Thomas_Gold_-_Pump_Up_The_Jam_71752208.mp3", "Thomas Gold", "Pump Up The Jam"),
        ("Vanotek_Denitia_-_Someone_72887479.mp3", "Vanotek", "Someone"),
    ]
    for (audio_file_name, track_artist, track_name) in tests:
        run_track(audio_file_name, track_artist, track_name,
                  num_samples=3,
                  output_dir="diploma_test/generated_tanh_xxx_2",
                  num_start=0, gen_canvas_size_=1024)


def test3():
    tests = [
        # ("Herzeloyde - Paroxysm.mp3", "Herzeloyde", "Paroxysm"),
        # ("George Michael - Carles Whisper.mp3", "George Michael", "Carles Whisper"),
        # ("24kGoldn_-_Love_Or_Lust_72874141.mp3", "24kGoldn", "Love Or Lust"),
        # ("Burak_Yeter_-_Body_Talks_71511893.mp3", "Burak Yeter", "Body Talks"),
        # ("Dotan_-_Numb_68724683.mp3", "Dotan", "Numb"),
        # ("Ed_Sheeran_-_Afterglow_72055475.mp3", "Ed Sheeran", "Afterglow"),
        # ("Foushee_-_Deep_End_70581187.mp3", "Foushee", "Deep End"),
        # ("Helena_Arash_-_Angels_Lullaby_72911174.mp3", "Arash", "Angels Lullaby"),
        # ("Imagine_Dragons_-_Follow_You_72848541 (1).mp3", "Imagine Dragons", "Follow You"),
        # ("Minelli_-_Rampampam_72874060.mp3", "Minelli", "Rampampam"),
        # ("Pnk_Willow_Sage_Hart_-_Cover_Me_In_Sunshine_72663179.mp3", "P!nk", "Cover Me In Sunshine"),
        # ("Rita_Ora_Imanbek_-_Bang_Bang_72665055.mp3", "Rita Ora", "Bang Bang"),
        # ("Riton_x_Nightcrawlers_-_Friday_feat_Mufasa_Hypeman_Dopamine_Re-edit_73265875.mp3",
        #  "Nightcrawlers", "Friday"),
        ("Nessa_Barrett_jxdn_-_la_di_die_72757984.mp3", "Nessa Barrett", "la di die"),
        ("Tesher_-_Jalebi_Baby_72865842.mp3", "Tesher", "Jalebi Baby"),
        ("Thomas_Gold_-_Pump_Up_The_Jam_71752208.mp3", "Thomas Gold", "Pump Up The Jam"),
        ("Vanotek_Denitia_-_Someone_72887479.mp3", "Vanotek", "Someone"),
    ]
    output_dir = "diploma_test/generated_emotions6"
    for (audio_file_name, track_artist, track_name) in tests:
        for emotion in Emotion:
            res_dir = f"{output_dir}/{str(emotion)}"
            os.makedirs(res_dir, exist_ok=True)
            run_track(audio_file_name, track_artist, track_name,
                      num_samples=1,
                      emotions=[str(emotion)],
                      output_dir=res_dir,
                      num_start=0, gen_canvas_size_=1024, deterministic=True)


def speed_test():
    output_dir = "diploma_test/generated_speeds5"
    os.makedirs(output_dir, exist_ok=True)
    all_fnames = os.listdir("diploma_test/speed_musics_ckpts_TS")
    tests = list(map(lambda it: fname_to_test(it), all_fnames))
    for (audio_file_name, track_artist, track_name) in tests:
        run_track(audio_file_name, track_artist, track_name,
                  num_samples=3,
                  output_dir=output_dir,
                  num_start=0, gen_canvas_size_=1024, deterministic=False)


def vol_test():
    output_dir = "diploma_test/generated_vols6"
    os.makedirs(output_dir, exist_ok=True)
    # all_fnames = os.listdir("diploma_test/vol_musics_ckpts2")
    # all_fnames = os.listdir("diploma_test/vol_music_ckpts_kino")
    all_fnames = os.listdir("diploma_test/vol_musics_LP_ckpts")
    tests = list(map(lambda it: fname_to_test(it), all_fnames))
    for (audio_file_name, track_artist, track_name) in tests:
        run_track(audio_file_name, track_artist, track_name,
                  num_samples=3,
                  output_dir=output_dir,
                  num_start=0, gen_canvas_size_=1024, deterministic=False)


def random_test():
    output_dir = "diploma_test/rand_music_generated_covers4"
    os.makedirs(output_dir, exist_ok=True)
    all_fnames = os.listdir("diploma_test/test_random_music/music_ckpts")
    tests = list(map(lambda it: fname_to_test(it), all_fnames))
    for (audio_file_name, track_artist, track_name) in tests[1:2]:
        run_track(audio_file_name, track_artist, track_name,
                  num_samples=3,
                  output_dir=output_dir,
                  num_start=0, gen_canvas_size_=1024, deterministic=False)


def random_test2():
    output_dir = "diploma_test/rand_music_generated_covers6"
    os.makedirs(output_dir, exist_ok=True)
    all_fnames = os.listdir("diploma_test/test_random_music/music_ckpts")
    tests = list(map(lambda it: fname_to_test(it), all_fnames))
    chosen_best = list(map(lambda it: fname_to_test(it[:-6] + ".mp3"),
                           os.listdir("diploma_test/chosen_best_covers")))
    for x in chosen_best:
        tests.remove(x)
    for (audio_file_name, track_artist, track_name) in chosen_best:
        choice = random.choice(tests)
        print(f"For `{track_artist} - {track_name}` randomly was chosen track: "
              f"`{choice[1]} - {choice[2]}`.")
        tests.remove(choice)
        run_track(choice[0], track_artist, track_name,
                  num_samples=3,
                  output_dir=output_dir,
                  num_start=0, gen_canvas_size_=1024, deterministic=False)


if __name__ == '__main__':
    test1()
    # test2()
    # test3()
    # speed_test()
    # vol_test()
    # random_test()
    # random_test2()
