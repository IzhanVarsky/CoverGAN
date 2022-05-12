import grpc

from .protosvg_pb2 import ProtoSVG
from .service_pb2 import ConvertRequest, RenderRequest
from .service_pb2_grpc import PSVGStub


CHANNEL_OPTIONS = [('grpc.lb_policy_name', 'pick_first'),
                   ('grpc.enable_retries', 0),
                   ('grpc.keepalive_timeout_ms', 10000)]


def color_to_int(r: int, g: int, b: int, a: int):
    return int.from_bytes([r, g, b, a], byteorder='little')


class PSVG:
    def __init__(self, target: str):
        self.channel = grpc.insecure_channel(target=target,
                                             options=CHANNEL_OPTIONS)
        self.stub = PSVGStub(self.channel)

    def convert_to_svg(self, svg: ProtoSVG) -> str:
        reply = self.stub.Convert(ConvertRequest(svg=svg), timeout=10)
        return reply.svg_xml

    def render(self, svg: ProtoSVG) -> bytes:
        reply = self.stub.Render(RenderRequest(svg=svg), timeout=10)
        return reply.png_data

    def convert_and_render(self, svg: ProtoSVG) -> (str, bytes):
        reply = self.stub.Render(RenderRequest(svg=svg), timeout=10)
        return reply.svg_xml, reply.png_data
