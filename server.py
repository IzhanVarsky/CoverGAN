import base64
import json
import logging
import mimetypes
import os
import random
import tempfile
import time

import magic
import psutil
import yaml

from multiprocessing import get_context, freeze_support

import cherrypy
from cherrypy import log

from outer.emotions import Emotion, emotion_from_str

SUPPORTED_EXTENSIONS = {"flac", "mp3", "aiff", "wav", "ogg"}

process = psutil.Process(os.getpid())  # For monitoring purposes

config = yaml.safe_load(open("config.yml"))

log_level = logging.getLevelName(config["app"]["log_level"])
logger = logging.getLogger("server")
logger.addHandler(logging.StreamHandler())
logger.setLevel(log_level)

html_header = \
    """
    <html>
        <head>
            <title>CoverGAN</title>
        </head>
        <body>
    """

html_footer = \
    """
        </body>
    </html>
    """

html_body = \
    """
    <p>
    <form action="generate" method="post" enctype="multipart/form-data">
        Audio file: <input type="file" name="audio_file">  <br> <br>
        Artist name: <input type="text" name="track_artist">  <br> <br>
        Track name: <input type="text" name="track_name">  <br> <br>
        Emotion: <select name="emotion">
                  <option selected>ANGER</option>
                <option>COMFORTABLE</option>
                <option>FEAR</option>
                <option>FUNNY</option>
                <option>HAPPY</option>
                <option>INSPIRATIONAL</option>
                <option>JOY</option>
                <option>LONELY</option>
                <option>NOSTALGIC</option>
                <option>PASSIONATE</option>
                <option>QUIET</option>
                <option>RELAXED</option>
                <option>ROMANTIC</option>
                <option>SADNESS</option>
                <option>SERIOUS</option>
                <option>SOULFUL</option>
                <option>SURPRISE</option>
                <option>SWEET</option>
                <option>WARY</option>
                 </select>  <br> <br>
        Generator type: <select name="gen_type">
                           <option>2</option>
                           <option>1</option>
                        </select>  <br> <br>
        Rasterize: <select name="rasterize">
                           <option>True</option>
                           <option>False</option>
                        </select>  <br> <br>
        Captioner type: <select name="use_captioner">
                           <option value="False">2</option>
                           <option value="True">1</option>
                        </select>  <br> <br>
        <input type="submit" value="Upload">
        </div>
    </form>
    </p>
    """


def base64_encode(img):
    return base64.b64encode(img).decode('utf-8')


def process_generate_request(tmp_filename: str,
                             track_artist: str, track_name: str,
                             emotions: [Emotion],
                             rasterize: bool,
                             gen_type: str,
                             use_captioner: bool) -> [(str, str)]:
    start = time.time()

    logger.info(f"REQ: artist={track_artist}, name={track_name}, emotions={emotions}, " +
                f"rasterize={rasterize}, gen_type={gen_type}, use_captioner={use_captioner}")

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
                                              rasterize, gen_type, use_captioner))
    os.remove(tmp_filename)
    if rasterize:
        result = list(map(lambda x: {"svg": x[0], "base64": base64_encode(x[1])}, result))
    else:
        result = list(map(lambda x: {"svg": x}, result))

    time_spent = time.time() - start
    log("Completed api call.Time spent {0:.3f} s".format(time_spent))

    return result


class ApiServerController(object):
    @cherrypy.expose('/health')
    def health(self):
        result = {
            "status": "OK",
            "info": {
                "mem": "{0:.3f} MiB".format(process.memory_info().rss / (1024 ** 2)),
                "cpu": process.cpu_percent(),
                "threads": len(process.threads())
            }
        }
        return json.dumps(result).encode("utf-8")

    @cherrypy.expose('/')
    def index(self):
        return html_header + html_body + html_footer

    @cherrypy.expose
    def generate(self, audio_file=None, track_artist: str = None, track_name: str = None, emotion: str = None,
                 rasterize=True, gen_type="2", use_captioner=False):
        print("audio_file:", audio_file)
        print("track_artist:", track_artist)
        print("track_name:", track_name)
        if audio_file is None or \
                audio_file.file is None or \
                track_artist is None or \
                track_name is None:
            raise cherrypy.HTTPRedirect("/")
            # return html_header + html_body + html_footer
        if rasterize == "True":
            rasterize = True
        else:
            rasterize = False
        if use_captioner == "True":
            use_captioner = True
        else:
            use_captioner = False
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

        generated = process_generate_request(
            tmp_filename,
            track_artist, track_name,
            [emotion], rasterize,
            gen_type, use_captioner
        )
        res_html = html_header
        res_html += "<h2>Generated SVG's:</h2>"
        for x in generated:
            res_html += x["svg"] + '\n'
        if rasterize:
            res_html += "<h2>Rasterized SVG's:</h2>"
            for x in generated:
                b64 = x["base64"]
                img_ = f'<img src="data:image/png;base64, {b64}" alt="rasterized img"/>'
                res_html += img_ + '\n'
        res_html += html_body
        res_html += html_footer
        return res_html


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
            ('Content-Type', 'text/html;encoding=utf-8')
        ],
    })

    from gen import do_generate

    process_pool = get_context("spawn").Pool(maxtasksperchild=20)

    try:
        cherrypy.engine.start()
        cherrypy.engine.block()
    except KeyboardInterrupt:
        cherrypy.engine.stop()
