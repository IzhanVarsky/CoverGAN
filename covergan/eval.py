#!/usr/bin/env python3
# coding: utf-8
import argparse
import logging
import os
from typing import Optional

from outer.emotions import Emotion, emotion_from_str
from service import CoverService, OverlayFilter

logger = logging.getLogger("eval")
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


def filter_from_str(filter_str: str) -> Optional[OverlayFilter]:
    try:
        return OverlayFilter[filter_str.upper()]
    except KeyError:
        print(f"Unknown filter: {filter_str}")
        return None


def main():
    parser = argparse.ArgumentParser()
    # Service config
    parser.add_argument("--gan1_weights", help="Model weights for CoverGAN",
                        type=str, default="./weights/covergan_ilya.pt")
    parser.add_argument("--gan2_weights", help="Model weights for CoverGAN",
                        type=str, default="./weights/checkpoint_6figs_5depth_512noise.pt")
    parser.add_argument("--captioner_weights", help="Captioner weights",
                        type=str, default="./weights/captioner.pt")
    parser.add_argument("--protosvg_address", help="ProtoSVG rendering server",
                        type=str, default="localhost:50051")
    parser.add_argument("--font_dir", help="Directory with font files",
                        type=str, default="./fonts")
    parser.add_argument("--output_dir", help="Directory where to save SVG covers",
                        type=str, default="./gen_samples")
    parser.add_argument("--num_samples", help="Number of samples to generate", type=int, default=5)
    # Input data
    parser.add_argument("--audio_file", help="Path to the audio file to process",
                        type=str, default=None, required=True)
    parser.add_argument("--emotions", help="Emotion of the audio file",
                        type=str, default=None, required=True)
    parser.add_argument("--track_artist", help="Track artist", type=str, default=None, required=True)
    parser.add_argument("--track_name", help="Track name", type=str, default=None, required=True)
    parser.add_argument("--gen_type", help="Type of generator to use (1 or 2)", type=str, default="2", required=True)
    parser.add_argument("--captioning_type", help="Type of captioning algo to use (1 or 2)", type=str,
                        default="2", required=True)
    # Other options
    parser.add_argument("--rasterize", help="Whether to rasterize the generated cover", default=False,
                        action="store_true")
    parser.add_argument("--filter", help="Overlay filter to apply to the final image",
                        default=False, action="store_true")
    parser.add_argument("--watermark", help="Whether to add watermark",
                        default=False, action="store_true")
    parser.add_argument("--debug", help="Whether to enable debug features",
                        default=False, action="store_true")
    parser.add_argument("--deterministic", help="Whether to disable random noise",
                        default=False, action="store_true")

    args = parser.parse_args()
    print(args)

    logger.info("--- Starting evaluator ---")

    # Validate the input
    audio_file_name = args.audio_file
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    emotions: [Emotion] = [emotion_from_str(x) for x in args.emotions.split(',')]

    if audio_file_name is None or None in emotions:
        print("ERROR: Missing audio/emotion, exiting.")
        return
    if args.track_artist is None or args.track_name is None:
        print("ERROR: Unspecified track authorship properties.")
        return
    if not os.path.isfile(audio_file_name):
        print("ERROR: The specified audio file does not exist.")
        return

    track_artist = args.track_artist
    track_name = args.track_name

    # Start the service
    service = CoverService(
        args.gan1_weights,
        args.captioner_weights,
        args.gan2_weights,
        args.font_dir,
        log_level=logging.INFO,
        debug=args.debug, deterministic=args.deterministic
    )

    # Generate covers
    result = service.generate(
        audio_file_name, track_artist, track_name, emotions,
        num_samples=args.num_samples, generatorType=args.gen_type, use_captioner=args.captioning_type,
        apply_filters=args.filter, rasterize=args.rasterize, watermark=args.watermark
    )

    basename = os.path.basename(audio_file_name)
    for i, res in enumerate(result):
        if args.rasterize:
            (svg_xml, png_data) = res
            svg_cover_filename = f"{output_dir}/{basename}-{i + 1}.svg"
            png_cover_filename = f"{output_dir}/{basename}-{i + 1}.png"
            with open(svg_cover_filename, 'w') as f:
                f.write(svg_xml)
            with open(png_cover_filename, 'wb') as f:
                f.write(png_data)
        else:
            svg_cover_filename = f"{output_dir}/{basename}-{i + 1}.svg"
            with open(svg_cover_filename, 'w') as f:
                f.write(res)


if __name__ == '__main__':
    main()
