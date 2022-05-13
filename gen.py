import logging
import yaml

from service import CoverService, GeneratorType

config = yaml.safe_load(open("config.yml"))

service = CoverService(
    config["service"]["protosvg_address"],
    config["service"]["gan_weights_ilya"],
    config["service"]["captioner_weights"],
    config["service"]["gan_weights_2"],
    config["service"]["font_dir"],
    log_level=logging.getLevelName(config["app"]["log_level"])
)


def do_generate(filename, track_artist, track_name, emotions, rasterize: bool,
                gen_type: str, use_captioner: bool):
    if gen_type == "1":
        gen_type = GeneratorType.IlyaGenerator
    elif gen_type == "2":
        gen_type = GeneratorType.GeneratorFixedSixPaths
    else:
        raise Exception(f"Unknown generator type: `{gen_type}`")
    return service.generate(
        filename, track_artist, track_name, emotions,
        num_samples=6, generatorType=gen_type, use_captioner=use_captioner,
        rasterize=rasterize, watermark=False
    )
