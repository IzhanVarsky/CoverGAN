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


def base64_encode(img):
    return base64.b64encode(img).decode('utf-8')


def process_generate_request(tmp_filename: str,
                             track_artist: str, track_name: str,
                             emotions: [Emotion]) -> [(str, str)]:
    start = time.time()

    logger.info(f"REQ: artist={track_artist}, name={track_name}, emotions={emotions}")

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

    rasterize = False

    # Execute the actual heavy computation in a process pool to escape GIL
    result = process_pool.apply(do_generate, (tmp_filename, track_artist, track_name, emotions, rasterize))
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

    @cherrypy.expose
    @cherrypy.tools.gzip()
    @cherrypy.tools.json_out()
    def generate(self, audio_file, track_artist: str, track_name: str, emotion: str = None):
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
            [emotion]
        )


if __name__ == '__main__':
    freeze_support()
    cherrypy.tree.mount(ApiServerController(), '/')

    cherrypy.init_func_types_config.update({
        'server.socket_port': config["app"]["port"],
        'server.socket_host': config["app"]["host"],
        'server.thread_pool': config["app"]["thread_pool"],
        'log.access_file': "access1.log",
        'log.error_file': "error1.log",
        'log.screen': True,
        'tools.response_headers.on': True,
        'tools.encode.encoding': 'utf-8',
        'tools.response_headers.headers': [
            ('Content-Type', 'application/json;encoding=utf-8')
        ],
    })

    from gen import do_generate

    process_pool = get_context("spawn").Pool(maxtasksperchild=20)

    try:
        cherrypy.engine.start()
        cherrypy.engine.block()
    except KeyboardInterrupt:
        cherrypy.engine.stop()
