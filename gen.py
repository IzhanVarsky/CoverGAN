import logging
import yaml

from service import CoverService

config = yaml.safe_load(open("config.yml"))

service = CoverService(
    config["service"]["protosvg_address"],
    config["service"]["gan_weights"],
    config["service"]["captioner_weights"],
    config["service"]["font_dir"],
    log_level=logging.getLevelName(config["app"]["log_level"])
)


def do_generate(filename, track_artist, track_name, emotions, rasterize):
    return service.generate(
        filename, track_artist, track_name, emotions,
        num_samples=6, rasterize=rasterize, watermark=False
    )
