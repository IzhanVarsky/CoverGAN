import io
import logging
from typing import Union

from captions.models.captioner import Captioner
from outer.audio_extractor import audio_to_embedding
from outer.emotions import Emotion, emotions_one_hot
from outer.models.generator import Generator
from outer.models.my_generator_fixed_six_figs import MyGeneratorFixedSixFigs
from outer.models.my_generator_fixed_six_figs_backup import MyGeneratorFixedSixFigs32
from outer.models.my_generator_rand_figure import MyGeneratorRandFigure
from outer.models.my_generator_three_figs import MyGeneratorFixedThreeFigs32
from service_utils import *
from utils.bboxes import BBox
from utils.noise import get_noise

logger = logging.getLogger("service")
logger.addHandler(logging.StreamHandler())


class GeneratorType(Enum):
    IlyaGenerator = 1
    GeneratorFixedSixPaths = 2
    GeneratorFixedThreeFigs32 = 3
    GeneratorFixedSixFigs32 = 4
    GeneratorRandFigure = 5


def image_to_byte_array(image: Image) -> bytes:
    imgByteArr = io.BytesIO()
    # image.save(imgByteArr, format=image.format)
    image.save(imgByteArr, format='PNG')
    return imgByteArr.getvalue()


def get_ilya_generator(gen_canvas_size_, weights, device, color_predictor_weights):
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
    return gen, z_dim_


def get_fixed_three_figs_32(gen_canvas_size_, weights, device, color_predictor_weights):
    z_dim = 32  # Dimension of the noise vector
    disc_slices_ = 6
    audio_embedding_dim = 281 * disc_slices_
    max_stroke_width = 0.01  # relative to the canvas size
    has_emotions = True
    gen = MyGeneratorFixedThreeFigs32(
        z_dim=z_dim,
        audio_embedding_dim=audio_embedding_dim,
        has_emotions=has_emotions,
        num_layers=5,
        canvas_size=gen_canvas_size_,
        path_count=-1,
        path_segment_count=-1,
        max_stroke_width=max_stroke_width,
        palette_model_weights=color_predictor_weights
    ).to(device)
    gen.eval()
    logger.info("INIT: Created fixed_three_figs_32 model.")
    gen_weights = torch.load(weights, map_location=device)
    gen.load_state_dict(gen_weights["0_state_dict"])
    logger.info("INIT: Loaded weights for fixed_three_figs_32 model.")
    return gen, z_dim


def get_fixed_six_paths(gen_canvas_size_, weights, device, color_predictor_weights):
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
    return gen, z_dim


def get_fixed_six_paths_32(gen_canvas_size_, weights, device, color_predictor_weights):
    z_dim = 32  # Dimension of the noise vector
    disc_slices_ = 6
    audio_embedding_dim = 281 * disc_slices_
    max_stroke_width = 0.01  # relative to the canvas size
    has_emotions = True
    gen = MyGeneratorFixedSixFigs32(
        z_dim=z_dim,
        audio_embedding_dim=audio_embedding_dim,
        has_emotions=has_emotions,
        num_layers=5,
        canvas_size=gen_canvas_size_,
        path_count=-1,
        path_segment_count=-1,
        max_stroke_width=max_stroke_width,
        palette_model_weights=color_predictor_weights
    ).to(device)
    gen.eval()
    logger.info("INIT: Created fixed_six_paths_32 model.")
    gen_weights = torch.load(weights, map_location=device)
    gen.load_state_dict(gen_weights["0_state_dict"])
    logger.info("INIT: Loaded weights for fixed_six_paths_32 model.")
    return gen, z_dim


def get_gen_rand_figs(gen_canvas_size_, weights, device, color_predictor_weights):
    z_dim = 512  # Dimension of the noise vector
    disc_slices_ = 6
    audio_embedding_dim = 281 * disc_slices_
    max_stroke_width = 0.01  # relative to the canvas size
    has_emotions = True
    gen = MyGeneratorRandFigure(
        z_dim=z_dim,
        audio_embedding_dim=audio_embedding_dim,
        has_emotions=has_emotions,
        num_layers=-1,
        canvas_size=gen_canvas_size_,
        path_count=-1,
        path_segment_count=-1,
        max_stroke_width=max_stroke_width,
        palette_model_weights=color_predictor_weights
    ).to(device)
    gen.eval()
    logger.info("INIT: Created gen_rand_figs model.")
    gen_weights = torch.load(weights, map_location=device)
    gen.load_state_dict(gen_weights["0_state_dict"])
    logger.info("INIT: Loaded weights for gen_rand_figs model.")
    return gen, z_dim


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


class CoverServiceForGenerator:
    def __init__(self, gen_type: GeneratorType,
                 gen_weights: str,
                 use_captioner: bool,
                 font_dir: str,
                 color_predictor_weights: str = None,
                 captioner_weights: str = None,
                 deterministic: bool = False, debug: bool = False,
                 log_level: int = logging.ERROR,
                 renderer_type='cairo'):
        assert os.path.isfile(gen_weights)
        assert os.path.isdir(font_dir)

        self.font_dir_ = font_dir
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.deterministic_ = deterministic
        self.debug_ = debug
        self.use_captioner = use_captioner
        self.gen_type = gen_type
        self.renderer_type = renderer_type
        logger.setLevel(log_level)

        self.disc_slices_ = 6
        self.gen_canvas_size_ = 512
        self.captioner_canvas_size_ = 256

        self.generator, self.gen_noise_zdim = to_func[gen_type](self.gen_canvas_size_, gen_weights,
                                                                self.device_, color_predictor_weights)

        if self.use_captioner:
            self.captioner = get_captioner(self.captioner_canvas_size_, captioner_weights, self.device_)

    def generate(self,
                 audio_file_name: str, track_artist: str, track_name: str,
                 emotions: [Emotion], num_samples: int,
                 use_color_predictor=False,
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

        logger.info(f"GEN: Generating the covers with {self.gen_type}...")
        noise = self.create_noise(num_samples, z_dim=self.gen_noise_zdim)
        psvg_covers = self.generator(noise, music_tensor, emotions_tensor, return_psvg=True)
        # use_triad_coloring?? use_palette??
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
        if self.use_captioner:
            logger.info("Using Captioner Model...")
            cover_render_pils = [
                png_data_to_pil_image(x, self.captioner_canvas_size_)
                for x in cover_render_pngs
            ]
            cover_render_tensors = torch.stack([to_tensor(x) for x in cover_render_pils]).to(self.device_)

            # Generate the captions
            logger.info("CAPT: Generating captions...")
            pos_preds, color_preds = self.captioner(cover_render_tensors)
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

                add_caption(
                    psvg_cover, self.font_dir_,
                    track_artist, artist_name_pos, artist_name_color,
                    track_name, track_name_pos, track_name_color,
                    debug=self.debug_, deterministic=self.deterministic_
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

        logger.info("PSVG: Making SVG+PNG...")
        if self.use_captioner:
            return list(map(lambda x: (str(x), x.render(renderer_type=self.renderer_type)), psvg_covers))
        #  return PILs images as rendered (ground truth)
        res = []
        for i, psvg in enumerate(psvg_covers):
            png_data = image_to_byte_array(pils_render[i])
            res.append((str(psvg), png_data))
        return res

    def create_noise(self, num_samples, z_dim):
        if self.deterministic_:
            noise = torch.zeros((num_samples, z_dim), device=self.device_)
        else:
            noise = get_noise(num_samples, z_dim, device=self.device_)
        return noise


to_func = {
    GeneratorType.IlyaGenerator: get_ilya_generator,
    GeneratorType.GeneratorFixedThreeFigs32: get_fixed_three_figs_32,
    GeneratorType.GeneratorFixedSixFigs32: get_fixed_six_paths_32,
    GeneratorType.GeneratorFixedSixPaths: get_fixed_six_paths,
    GeneratorType.GeneratorRandFigure: get_gen_rand_figs,
}
