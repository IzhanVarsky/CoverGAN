import io
import logging
from typing import Union

import cairosvg
import numpy as np
from PIL import ImageDraw

from captions.models.captioner import Captioner
from fonts_cfg import FONTS
from outer.audio_extractor import audio_to_embedding
from outer.emotions import Emotion, emotions_one_hot
from outer.models.generator import Generator
from outer.models.my_generator_fixed_six_figs import MyGeneratorFixedSixFigs
from service_utils import *
from utils.bboxes import BBox
from utils.deterministic_text_fitter import get_all_boxes_info_to_paste, draw_to_draw_object
from utils.noise import get_noise

logger = logging.getLogger("service")
logger.addHandler(logging.StreamHandler())


class GeneratorType(Enum):
    IlyaGenerator = 1
    GeneratorFixedSixPaths = 2


def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    # image.save(imgByteArr, format=image.format)
    image.save(imgByteArr, format='PNG')
    return imgByteArr.getvalue()


def get_ilya_generator(gen_canvas_size_, weights, device):
    z_dim_ = 32  # Dimension of the noise vector
    num_gen_layers = 5
    disc_slices_ = 6
    audio_embedding_dim = 281 * disc_slices_
    path_count = 3
    path_segment_count = 4
    max_stroke_width = 0.01  # relative to the canvas size
    has_emotions = True
    gen = Generator(
        z_dim=z_dim_,
        audio_embedding_dim=audio_embedding_dim,
        has_emotions=has_emotions,
        num_layers=num_gen_layers,
        canvas_size=gen_canvas_size_,
        path_count=path_count,
        path_segment_count=path_segment_count,
        max_stroke_width=max_stroke_width
    ).to(device)
    gen.eval()
    logger.info("INIT: Created ilya_generator model.")
    gen_weights = torch.load(weights, map_location=device)
    gen.load_state_dict(gen_weights["0_state_dict"])
    logger.info("INIT: Loaded weights for ilya_generator model.")
    return gen


def get_fixed_six_paths(gen_canvas_size_, weights, device):
    z_dim = 512  # Dimension of the noise vector
    disc_slices_ = 6
    audio_embedding_dim = 281 * disc_slices_
    max_stroke_width = 0.01  # relative to the canvas size
    has_emotions = True
    gen = MyGeneratorFixedSixFigs(
        z_dim=z_dim,
        audio_embedding_dim=audio_embedding_dim,
        has_emotions=has_emotions,
        num_layers=-1,
        canvas_size=gen_canvas_size_,
        path_count=-1,
        path_segment_count=-1,
        max_stroke_width=max_stroke_width
    ).to(device)
    gen.eval()
    logger.info("INIT: Created fixed_six_paths_generator model.")
    gen_weights = torch.load(weights, map_location=device)
    gen.load_state_dict(gen_weights["0_state_dict"])
    logger.info("INIT: Loaded weights for fixed_six_paths_generator model.")
    return gen


def get_captioner(captioner_canvas_size_, weights, device):
    num_captioner_conv_layers = 3
    num_captioner_linear_layers = 2
    captioner = Captioner(
        canvas_size=captioner_canvas_size_,
        num_conv_layers=num_captioner_conv_layers,
        num_linear_layers=num_captioner_linear_layers
    ).to(device)
    captioner.eval()
    logger.info("INIT: Created captioner model.")
    captioner_weights = torch.load(weights, map_location=device)
    captioner.load_state_dict(captioner_weights["0_state_dict"])
    logger.info("INIT: Loaded weights for captioner model.")
    return captioner


class CoverService:
    def __init__(self, gan_weights_ilya: str,
                 captioner_weights: str,
                 gan_weights_2: str,
                 font_dir: str,
                 deterministic: bool = False, debug: bool = False,
                 log_level: int = logging.ERROR):
        assert os.path.isfile(gan_weights_ilya)
        assert os.path.isfile(captioner_weights)
        assert os.path.isfile(gan_weights_2)
        assert os.path.isdir(font_dir)

        self.font_dir_ = font_dir
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.deterministic_ = deterministic
        self.debug_ = debug
        logger.setLevel(log_level)

        # Architecture properties
        self.z_dim_ = 32  # Dimension of the noise vector
        self.disc_slices_ = 6
        self.gen_canvas_size_ = 512
        self.captioner_canvas_size_ = 256

        self.ilya_generator = get_ilya_generator(self.gen_canvas_size_, gan_weights_ilya, self.device_)
        self.generator2 = get_fixed_six_paths(self.gen_canvas_size_, gan_weights_2, self.device_)
        self.captioner_ = get_captioner(self.captioner_canvas_size_, captioner_weights, self.device_)

    def generate(self,
                 audio_file_name: str, track_artist: str, track_name: str,
                 emotions: [Emotion], num_samples: int, generatorType: GeneratorType,
                 use_captioner: bool,
                 rasterize=False,
                 apply_filters=True, filtered_samples=None, watermark=False
                 ) -> Union[List[str], List[Tuple[str, bytes]]]:
        # logger.info(f"!!!! USE CAPTIONER: {use_captioner}")
        logger.info("GEN: Creating audio embedding...")
        # Prepare the input
        music_tensor = torch.from_numpy(audio_to_embedding(audio_file_name))
        target_count = 24  # 2m = 120s, 120/5
        if len(music_tensor) < target_count:
            music_tensor = music_tensor.repeat(target_count // len(music_tensor) + 1, 1)
        music_tensor = music_tensor[:target_count].float().to(self.device_)
        music_tensor = music_tensor[:self.disc_slices_].flatten()
        music_tensor = music_tensor.unsqueeze(dim=0).repeat((num_samples, 1))
        logger.info("GEN: Audio embedding done.")
        emotions_tensor = emotions_one_hot(emotions).to(self.device_).unsqueeze(dim=0).repeat((num_samples, 1))

        # Generate the covers
        if generatorType == GeneratorType.IlyaGenerator:
            logger.info("GEN: Generating the covers with IlyaGenerator...")
            ilya_noise = self.create_noise(num_samples, z_dim=32)
            psvg_covers = self.ilya_generator(ilya_noise, music_tensor, emotions_tensor,
                                                               return_psvg=True)
        elif generatorType == GeneratorType.GeneratorFixedSixPaths:
            logger.info("GEN: Generating the covers with GeneratorFixedSixPaths...")
            gen2_noise = self.create_noise(num_samples, z_dim=512)
            psvg_covers = self.generator2(gen2_noise, music_tensor, emotions_tensor,
                                          return_psvg=True,
                                          return_diffvg_svg_params=False,
                                          use_triad_coloring=False)
        else:
            msg = f"Generator type `{generatorType}` is not supported :("
            logger.error(msg)
            raise Exception(msg)
        logger.info("GEN: Covers generated.")

        # Apply overlay filters, if requested
        if apply_filters:
            logger.info("GEN: Will apply filters.")
            if filtered_samples is None:
                # filtered_samples = round(num_samples // 2)
                filtered_samples = num_samples
            filters = apply_filters if isinstance(apply_filters, list) else list(OverlayFilter)
            for psvg_cover in psvg_covers[-filtered_samples:]:
                overlay_filter = random.choice(filters)
                add_filter(psvg_cover, overlay_filter)

        logger.info("PSVG: Rendering backgrounds...")
        cover_render_pngs = [x.render(renderer_type="cairo") for x in psvg_covers]
        logger.info("PSVG: Backgrounds rendered.")
        if use_captioner:
            logger.info("Using Captioner Model...")
            cover_render_pils = [
                png_data_to_pil_image(x, self.captioner_canvas_size_)
                for x in cover_render_pngs
            ]
            cover_render_tensors = torch.stack([to_tensor(x) for x in cover_render_pils]).to(self.device_)

            # Generate the captions
            logger.info("CAPT: Generating captions...")
            pos_preds, color_preds = self.captioner_(cover_render_tensors)
            logger.info("CAPT: Captions generated.")
            pos_preds = torch.round(pos_preds * self.gen_canvas_size_).to(int)
            color_preds = torch.round(color_preds * 255).to(int)
            artist_name_positions, track_name_positions = pos_preds[:, :4], pos_preds[:, 4:]
            artist_name_colors, track_name_colors = color_preds[:, :3], color_preds[:, 3:]

            # Combine the covers with the captions
            logger.info("FIT: Fitting the captions...")
            for i in range(num_samples):
                psvg_cover = psvg_covers[i]
                pil_cover: Image.Image = cover_render_pils[i]
                artist_name_pos: BBox = pos_tensor_to_bbox(artist_name_positions[i])
                track_name_pos: BBox = pos_tensor_to_bbox(track_name_positions[i])
                artist_name_color: Tuple[int, int, int] = tuple(map(int, artist_name_colors[i]))
                track_name_color: Tuple[int, int, int] = tuple(map(int, track_name_colors[i]))

                artist_name_color = ensure_caption_contrast(
                    pil_cover,
                    artist_name_pos.recanvas(self.gen_canvas_size_, self.captioner_canvas_size_),
                    artist_name_color
                )
                track_name_color = ensure_caption_contrast(
                    pil_cover,
                    track_name_pos.recanvas(self.gen_canvas_size_, self.captioner_canvas_size_),
                    track_name_color
                )

                # Font properties
                artist_font_family = FONTS[0] if self.deterministic_ else random.choice(FONTS)
                name_font_family = FONTS[0] if self.deterministic_ else random.choice(FONTS)

                add_caption(
                    psvg_cover, self.font_dir_,
                    track_artist, artist_name_pos, artist_name_color, artist_font_family,
                    track_name, track_name_pos, track_name_color, name_font_family,
                    debug=self.debug_
                )
        else:
            logger.info("Using deterministic caption algo...")
            pils_render = [
                png_data_to_pil_image(x, self.gen_canvas_size_)
                for x in cover_render_pngs
            ]
            for i in range(num_samples):
                psvg_cover = psvg_covers[i]
                pil_img = pils_render[i]
                paste_caption(psvg_cover, pil_img, track_artist, track_name, self.font_dir_, for_svg=True)
        logger.info("FIT: Captions fitted.")

        # Add a service watermark if requested+
        if watermark:
            for psvg_cover in psvg_covers:
                add_watermark(psvg_cover)

        if rasterize:
            # [(svg_xml: str, png_data: bytes)]
            output_func = lambda x: (str(x), x.render(renderer_type="cairo"))
            logger.info("PSVG: Making SVG+PNG...")
        else:
            # [svg_xml: str]
            output_func = str
            logger.info("PSVG: Making SVG...")
        result = list(map(output_func, psvg_covers))

        new_res = []
        for i, res in enumerate(result):
            if rasterize:
                (svg_xml, png_data) = res
                # svg_xml = add_font_imports_to_svg_xml(svg_xml)
                if not use_captioner:
                    png_data = image_to_byte_array(pils_render[i])
                else:
                    png_data = cairosvg.svg2png(bytestring=svg_xml.encode("utf-8"))
                new_res.append((svg_xml, png_data))
            else:
                # new_res.append(add_font_imports_to_svg_xml(res))
                new_res.append(res)
        return new_res

    def create_noise(self, num_samples, z_dim):
        if self.deterministic_:
            noise = torch.zeros((num_samples, z_dim), device=self.device_)
        else:
            noise = get_noise(num_samples, z_dim, device=self.device_)
        return noise
