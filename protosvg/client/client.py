#!/usr/bin/env python3
from io import BytesIO
import logging
from PIL import Image

import grpc

import protosvg_pb2
import service_pb2
import service_pb2_grpc


CHANNEL_OPTIONS = [('grpc.lb_policy_name', 'pick_first'),
                   ('grpc.enable_retries', 0),
                   ('grpc.keepalive_timeout_ms', 10000)]


def color_to_int(r: int, g: int, b: int, a: int):
    return int.from_bytes([r, g, b, a], byteorder='little')


class PSVG:
    def __init__(self, target: str):
        self.channel = grpc.insecure_channel(target=target,
                                             options=CHANNEL_OPTIONS)
        self.stub = service_pb2_grpc.PSVGStub(self.channel)

    def convert_to_svg(self, svg: protosvg_pb2.ProtoSVG) -> str:
        reply = self.stub.Convert(service_pb2.ConvertRequest(svg=svg),
                                  timeout=10)
        return reply.svg_xml

    def render(self, svg: protosvg_pb2.ProtoSVG) -> bytes:
        reply = self.stub.Render(service_pb2.RenderRequest(svg=svg),
                                 timeout=10)
        return reply.png_data


if __name__ == '__main__':
    logging.basicConfig()
    psvg = PSVG('localhost:50051')

    svg = protosvg_pb2.ProtoSVG()
    svg.width = 1200
    svg.height = 1200
    svg.backgroundColor.rgba = color_to_int(r=29, g=154, b=243, a=255)

    svg_xml = psvg.convert_to_svg(svg)
    print(svg_xml)

    png_data = psvg.render(svg)
    image = Image.open(BytesIO(png_data))
    image.thumbnail((500, 500), Image.ANTIALIAS)
    Image.composite(image, Image.new('RGB', image.size, 'white'), image).show()
