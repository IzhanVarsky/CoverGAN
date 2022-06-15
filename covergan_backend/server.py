import base64
import logging
import mimetypes
import os
import random
import tempfile
import time
from multiprocessing import get_context, freeze_support

import cherrypy
import magic
import psutil
import yaml
from cherrypy import log

from outer.emotions import Emotion, emotion_from_str

SUPPORTED_EXTENSIONS = {"flac", "mp3", "aiff", "wav", "ogg"}

process = psutil.Process(os.getpid())  # For monitoring purposes

config = yaml.safe_load(open("config.yml"))

fonts_folder = config["service"]["font_dir"]

log_level = logging.getLevelName(config["app"]["log_level"])
logger = logging.getLogger("server")
logger.addHandler(logging.StreamHandler())
logger.setLevel(log_level)


def base64_encode(img):
    return base64.b64encode(img).decode('utf-8')


def process_generate_request(tmp_filename: str,
                             track_artist: str, track_name: str,
                             emotions: [Emotion],
                             gen_type: str,
                             use_captioner: bool,
                             num_samples: int, use_filters: bool):
    start = time.time()

    logger.info(f"REQ: artist={track_artist}, name={track_name}, emotions={emotions}, " +
                f"gen_type={gen_type}, use_captioner={use_captioner}")

    mime = magic.Magic(mime=True)
    ext = mimetypes.guess_extension(mime.from_file(tmp_filename))
    if ext is None:
        os.remove(tmp_filename)
        logger.info("REQ: Rejecting, unrecognized file format")
        raise cherrypy.HTTPError(400, message="Unrecognized file format")
    elif ext[1:] not in SUPPORTED_EXTENSIONS:
        os.remove(tmp_filename)
        logger.info(f"REQ: Rejecting, unsupported file format: {ext}")
        raise cherrypy.HTTPError(400, message="Unsupported file format")
    else:
        logger.info(f"REQ: Accepting file format: {ext}")
        os.rename(tmp_filename, tmp_filename + ext)
        tmp_filename += ext

    # Execute the actual heavy computation in a process pool to escape GIL
    result = process_pool.apply(do_generate, (tmp_filename, track_artist, track_name, emotions,
                                              gen_type, use_captioner, num_samples, use_filters))
    os.remove(tmp_filename)
    result = list(map(lambda x: {"svg": x[0], "base64": base64_encode(x[1])}, result))

    time_spent = time.time() - start
    log("Completed api call.Time spent {0:.3f} s".format(time_spent))

    return {"result": result}


def str_to_bool(s: str):
    return True if s.upper() == "True".upper() else False


class ApiServerController(object):
    @cherrypy.expose('/health')
    @cherrypy.tools.json_out()
    def health(self):
        return {
            "status": "OK",
            "info": {
                "mem": "{0:.3f} MiB".format(process.memory_info().rss / (1024 ** 2)),
                "cpu": process.cpu_percent(),
                "threads": len(process.threads())
            }
        }

    @cherrypy.expose('/get_emotions')
    @cherrypy.tools.json_out()
    def get_emotions(self):
        return {"emotions": [x.name for x in Emotion]}

    @cherrypy.expose('/')
    def index(self):
        return open("html/index.html")

    @cherrypy.expose('/style')
    def style(self):
        return open("html/style-transfer.html")

    @cherrypy.expose('/raster')
    def raster(self):
        return open("html/rasterize.html")

    @cherrypy.expose('/color-extractor')
    def color_extractor(self):
        return open("html/color-extractor.html")

    @cherrypy.expose('/text_paste')
    def text_paste(self):
        return open("html/text-paste.html")

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def svg_to_json(self, svg=None):
        print("svg:", svg)
        if svg is None:
            raise cherrypy.HTTPRedirect("/")
        from outer.SVGContainer import SVGContainer
        res_svg_json = SVGContainer.load_svg(svg).to_obj()
        return {"result": res_svg_json}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def add_text_svg(self, svg_img=None, artist_name="", track_name=""):
        from outer.SVGContainer import SVGContainer
        from service_utils import paste_caption
        from io import BytesIO
        svg_cont = SVGContainer.load_svg(svg_img)
        pil = svg_cont.to_PIL(renderer_type="cairo").convert("RGB")
        paste_caption(svg_cont, pil, artist_name, track_name, font_dir=fonts_folder)
        buffered = BytesIO()
        pil.save(buffered, format="PNG")
        return {"result": {
            "svg_before": svg_img,
            "svg": str(svg_cont),
            "png": base64_encode(buffered.getvalue())
        }}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def extract_colors(self, img=None, color_count=None, algo_type="1", use_random: str = False):
        color_count = int(color_count)
        use_random = str_to_bool(use_random)
        from colorer.music_palette_dataset import get_main_rgb_palette, get_main_rgb_palette2
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_filename = f.name
            while True:
                data = img.file.read()
                if not data:
                    break
                f.write(data)
        if algo_type == "1":
            palette = get_main_rgb_palette(tmp_filename, color_count)
        else:
            palette = get_main_rgb_palette2(tmp_filename, color_count)
        print("Palette:", palette)
        if use_random:
            random.shuffle(palette)
        return {"result": [list(map(int, x)) for x in palette]}

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def rasterize(self, svg=None):
        print("svg:", svg)
        if svg is None:
            raise cherrypy.HTTPRedirect("/")
        from outer.SVGContainer import cairo_rendering, wand_rendering, svglib_rendering
        return {"result": {
            "svg": svg,
            "res_png1": base64_encode(cairo_rendering(svg)),
            # "res_png2": base64_encode(wand_rendering(svg)),
            # "res_png3": base64_encode(svglib_rendering(svg)),
        }
        }

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def style_transfer(self, img_from=None, svg_to=None):
        print("img_from:", img_from)
        print("svg_to:", svg_to)
        if img_from.file is None or svg_to is None:
            raise cherrypy.HTTPRedirect("/")
        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_filename = f.name
            while True:
                data = img_from.file.read()
                if not data:
                    break
                f.write(data)
        from colorer.transfer_style import transfer_style_str
        res = {"result": {
            "img_from": base64_encode(open(tmp_filename, "rb").read()),
            "svg_to": svg_to,
            "res_svg": transfer_style_str(tmp_filename, svg_to),
        }
        }
        os.remove(tmp_filename)
        return res

    @cherrypy.expose
    @cherrypy.tools.json_out()
    def generate(self, audio_file=None, track_artist: str = None,
                 track_name: str = None, emotion: str = None,
                 gen_type="2", use_captioner: str = False,
                 num_samples=5, use_filters: str = False):
        # TODO: Add colorer using
        print("audio_file:", audio_file)
        print("track_artist:", track_artist)
        print("track_name:", track_name)
        print("emotion:", emotion)
        print("gen_type:", gen_type)
        print("use_captioner:", use_captioner)
        print("num_samples:", num_samples)
        print("use_filters:", use_filters)
        if audio_file is None or \
                audio_file.file is None or \
                track_artist is None or \
                track_name is None:
            raise cherrypy.HTTPRedirect("/")
        use_captioner = str_to_bool(use_captioner)
        use_filters = str_to_bool(use_filters)
        num_samples = int(num_samples)
        if emotion is None:
            emotion = random.choice(list(Emotion))
        else:
            emotion = emotion_from_str(emotion)
            if emotion is None:
                raise cherrypy.HTTPError(400, message="Incorrect emotion specified")

        track_artist = track_artist[:50]
        track_name = track_name[:70]

        with tempfile.NamedTemporaryFile(delete=False) as f:
            tmp_filename = f.name
            while True:
                data = audio_file.file.read(8192)
                if not data:
                    break
                f.write(data)

        return process_generate_request(
            tmp_filename,
            track_artist, track_name,
            [emotion],
            gen_type, use_captioner,
            num_samples, use_filters
        )


if __name__ == '__main__':
    freeze_support()
    cherrypy.tree.mount(ApiServerController(), '/')

    cherrypy.config.update({
        'server.socket_port': config["app"]["port"],
        'server.socket_host': config["app"]["host"],
        'server.thread_pool': config["app"]["thread_pool"],
        'log.access_file': "access1.log",
        'log.error_file': "error1.log",
        'log.screen': True,
        'tools.response_headers.on': True,
        'tools.encode.encoding': 'utf-8',
        'tools.response_headers.headers': [
            ('Content-Type', 'text/html;encoding=utf-8'),
            ("Access-Control-Allow-Origin", "*"),
        ],
    })

    from gen import do_generate

    process_pool = get_context("spawn").Pool(maxtasksperchild=20)

    try:
        cherrypy.engine.start()
        cherrypy.engine.block()
    except KeyboardInterrupt:
        cherrypy.engine.stop()
