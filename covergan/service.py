import logging
import math
from enum import Enum
from typing import List, Union

from captions.models.captioner import Captioner
from fonts_cfg import FONTS
from outer.audio_extractor import audio_to_embedding
from outer.emotions import Emotion, emotions_one_hot
from outer.models.generator import Generator
from protosvg.client import PSVG
from service_utils import *
from utils.bboxes import BBox
from utils.noise import get_noise

logger = logging.getLogger("service")
logger.addHandler(logging.StreamHandler())

SERVICE_NAME = "PROTECTED"


class OverlayFilter(Enum):
    DIM = 0
    VIGNETTE = 1
    VINTAGE = 2
    SEPIA = 3
    WHITEN = 4


def add_filter(image: psvg.ProtoSVG, f: OverlayFilter):
    overlay = psvg.Square()
    overlay.start.x = 0
    overlay.start.y = 0
    overlay.width = image.width
    s = image.shapes.add()
    s.square.CopyFrom(overlay)

    if f == OverlayFilter.DIM:
        s.color.rgba = color_to_int(r=0, g=0, b=0, a=100)
    elif f == OverlayFilter.VINTAGE:
        s.color.rgba = color_to_int(r=255, g=244, b=226, a=100)
    elif f == OverlayFilter.SEPIA:
        s.color.rgba = color_to_int(r=173, g=129, b=66, a=100)
    elif f == OverlayFilter.WHITEN:
        s.color.rgba = color_to_int(r=255, g=255, b=255, a=100)
    elif f == OverlayFilter.VIGNETTE:
        gradient = psvg.RadialGradient()

        # Center
        stop = gradient.base.stops.add()
        stop.offset = 0
        stop.color.rgba = color_to_int(r=0, g=0, b=0, a=0)
        # Circular start
        stop = gradient.base.stops.add()
        stop.offset = 0.7
        stop.color.rgba = color_to_int(r=0, g=0, b=0, a=0)
        # End
        stop = gradient.base.stops.add()
        stop.offset = 1.0
        stop.color.rgba = color_to_int(r=0, g=0, b=0, a=153)

        gradient.end.x = image.width // 2
        gradient.end.y = image.height // 2
        gradient.endRadius = math.ceil(image.width / math.sqrt(2))
        s.radialGradient.CopyFrom(gradient)


def add_watermark(image: psvg.ProtoSVG):
    label = psvg.Label()

    label.font.size = 40
    label.font.family = "Roboto"
    label.font.weight = 700

    label.text = SERVICE_NAME
    label.start.x = image.width - 300
    label.start.y = image.height - 30

    s = image.shapes.add()
    s.label.CopyFrom(label)
    s.color.rgba = color_to_int(r=255, g=255, b=255, a=100)


class CoverService:
    def __init__(self, protosvg_address: str, gan_weights: str, captioner_weights: str, font_dir: str,
                 deterministic: bool = False, debug: bool = False, log_level: int = logging.ERROR):
        assert os.path.isfile(gan_weights)
        assert os.path.isfile(captioner_weights)
        assert os.path.isdir(font_dir)

        self.font_dir_ = font_dir
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.psvg_client_ = PSVG(protosvg_address)
        self.deterministic_ = deterministic
        self.debug_ = debug
        logger.setLevel(log_level)

        # Architecture properties
        self.z_dim_ = 32  # Dimension of the noise vector
        num_gen_layers = 5
        num_captioner_conv_layers = 3
        num_captioner_linear_layers = 2
        self.disc_slices_ = 6
        audio_embedding_dim = 281 * self.disc_slices_

        # Painter properties
        path_count = 3
        path_segment_count = 4
        max_stroke_width = 0.01  # relative to the canvas size

        # Training properties
        has_emotions = True
        self.captioner_canvas_size_ = 256

        # Eval properties
        self.gen_canvas_size_ = 512

        # Instantiate the models
        self.generator_ = Generator(
            z_dim=self.z_dim_,
            audio_embedding_dim=audio_embedding_dim,
            has_emotions=has_emotions,
            num_layers=num_gen_layers,
            canvas_size=self.gen_canvas_size_,
            path_count=path_count,
            path_segment_count=path_segment_count,
            max_stroke_width=max_stroke_width
        ).to(self.device_)
        self.generator_.eval()
        self.captioner_ = Captioner(
            canvas_size=self.captioner_canvas_size_,
            num_conv_layers=num_captioner_conv_layers,
            num_linear_layers=num_captioner_linear_layers
        ).to(self.device_)
        self.captioner_.eval()

        logger.info("INIT: Created the models.")

        # Load model weights
        gan_weights = torch.load(gan_weights, map_location=self.device_)
        self.generator_.load_state_dict(gan_weights["0_state_dict"])
        captioner_weights = torch.load(captioner_weights, map_location=self.device_)
        self.captioner_.load_state_dict(captioner_weights["0_state_dict"])
        logger.info("INIT: Loaded model weights.")

    def generate(self,
                 audio_file_name: str, track_artist: str, track_name: str,
                 emotions: [Emotion], num_samples: int, rasterize=False,
                 apply_filters=True, filtered_samples=None, watermark=False
                 ) -> Union[List[str], List[Tuple[str, bytes]]]:
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
        if self.deterministic_:
            noise = torch.zeros((num_samples, self.z_dim_), device=self.device_)
        else:
            noise = get_noise(num_samples, self.z_dim_, device=self.device_)

        # Generate the covers
        logger.info("GEN: Generating the covers...")
        psvg_covers: [psvg.ProtoSVG] = self.generator_(noise, music_tensor, emotions_tensor, return_psvg=True)
        logger.info("GEN: Covers generated.")

        # Apply overlay filters, if requested
        if apply_filters:
            logger.info("GEN: Will apply filters.")
            if filtered_samples is None:
                filtered_samples = round(num_samples // 2)
            filters = apply_filters if isinstance(apply_filters, list) else list(OverlayFilter)
            for psvg_cover in psvg_covers[-filtered_samples:]:
                overlay_filter = random.choice(filters)
                add_filter(psvg_cover, overlay_filter)

        logger.info("PSVG: Rendering backgrounds...")
        cover_render_pngs = [self.psvg_client_.render(x) for x in psvg_covers]
        logger.info("PSVG: Backgrounds rendered.")
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
            psvg_cover: psvg.ProtoSVG = psvg_covers[i]
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
                track_artist, artist_name_pos, rgb_tuple_to_int(artist_name_color), artist_font_family,
                track_name, track_name_pos, rgb_tuple_to_int(track_name_color), name_font_family,
                debug=self.debug_
            )
        logger.info("FIT: Captions fitted.")

        # Add a service watermark if requested+
        if watermark:
            for psvg_cover in psvg_covers:
                add_watermark(psvg_cover)

        if rasterize:
            # [(svg_xml: str, png_data: bytes)]
            output_func = self.psvg_client_.convert_and_render
            logger.info("PSVG: Making SVG+PNG...")
        else:
            # [svg_xml: str]
            output_func = self.psvg_client_.convert_to_svg
            logger.info("PSVG: Making SVG...")
        return list(map(output_func, psvg_covers))
