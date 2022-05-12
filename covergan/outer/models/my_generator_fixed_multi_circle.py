import torch
from torch import nn
from typing import *
from colorer.test_model import get_palette_predictor
from .former.modules import TransformerBlock
from .my_generator_circle_paths import create_circle_control_points
from ..emotions import Emotion
from ..represent import *


class MyGeneratorFixedMultiCircle(nn.Module):
    def __init__(self, z_dim: int, audio_embedding_dim: int, has_emotions: bool, num_layers: int, canvas_size: int,
                 path_count: int, path_segment_count: int, max_stroke_width: float):
        super(MyGeneratorFixedMultiCircle, self).__init__()
        self.path_count_in_row = 15
        path_count = self.path_count_in_row ** 2
        path_segment_count = 2

        in_features = z_dim + audio_embedding_dim
        if has_emotions:
            in_features += len(Emotion)

        self.circle_center_count = 2  # (x_0, y_0)
        self.circle_radius_count = 1
        self.background_color_count = 3  # RGB, no transparency for the background
        self.all_paths_count = path_count

        self.WITH_DEFORMATION = False
        if self.WITH_DEFORMATION:
            self.one_path_points_count = (path_segment_count * 3) * 2
            self.all_points_count_for_path = self.circle_radius_count + self.one_path_points_count
        else:
            self.all_points_count_for_path = self.circle_radius_count

        self.WITH_OPACITY = False
        if self.WITH_OPACITY:
            # 4 = RGBA
            self.fill_color = 4
        else:
            self.fill_color = 3
        self.all_points_count_for_path += self.fill_color
        out_dim = self.background_color_count + self.all_paths_count * self.all_points_count_for_path

        out_features = out_dim
        feature_step = (in_features - out_features) // num_layers

        layers = []
        for i in range(num_layers - 1):
            out_features = in_features - feature_step
            layers += [
                torch.nn.Linear(in_features=in_features, out_features=out_features),
                torch.nn.BatchNorm1d(num_features=out_features),
                torch.nn.LeakyReLU(0.2)
            ]
            in_features = out_features
        layers += [
            torch.nn.Linear(in_features=in_features, out_features=out_dim),
            torch.nn.Sigmoid()
        ]

        # self.transformer_block = TransformerBlock(1, 10, False)
        self.model_ = torch.nn.Sequential(*layers)

        self.canvas_size_ = canvas_size
        self.path_segment_count_ = path_segment_count
        self.max_stroke_width_ = max_stroke_width

    def forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                emotions: Optional[torch.Tensor], return_psvg=False) \
            -> Union[torch.Tensor, List[psvg.ProtoSVG]]:
        forward_fun = self.my_mega_forward
        return forward_fun(noise, audio_embedding, emotions, return_psvg)

    def my_mega_forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                        emotions: Optional[torch.Tensor], return_psvg=False, return_diffvg_svg_params=False) \
            -> Union[torch.Tensor, List[psvg.ProtoSVG]]:
        if emotions is not None:
            inp = torch.cat((noise, audio_embedding, emotions), dim=1)
        else:
            inp = torch.cat((noise, audio_embedding), dim=1)

        # inp = self.transformer_block(inp.view(inp.shape[0], -1, 1))
        # inp = inp.view(inp.shape[0], -1)
        all_shape_params = self.model_(inp)
        assert not torch.any(torch.isnan(all_shape_params))

        action = as_protosvg if return_psvg else as_diffvg_render

        result = []
        result_svg_params = []
        for b_idx, shape_params in enumerate(all_shape_params):
            index = 0

            inc = self.background_color_count
            background_color = shape_params[index: index + inc]
            index += inc

            paths = []
            for path_idx in range(self.all_paths_count):
                path = {}
                base_radius = self.canvas_size_ / (self.path_count_in_row * 2)

                inc = self.circle_radius_count
                radius = shape_params[index: index + inc] * base_radius * 2.5  # 1.5
                index += inc

                centerX = (path_idx % self.path_count_in_row) * (2 * base_radius) + base_radius
                centerY = (path_idx // self.path_count_in_row) * (2 * base_radius) + base_radius
                center_point = torch.tensor([centerX, centerY]).to(radius.device)
                circle_points = create_circle_control_points(center_point, radius, self.path_segment_count_)

                if self.WITH_DEFORMATION:
                    inc = self.one_path_points_count
                    # deformation_points = (shape_params[index: index + inc] - 0.5) * 2 * self.canvas_size_ * 0.1
                    deformation_points = (shape_params[index: index + inc] - 0.5) * 2 * radius * 0.2
                    index += inc
                    deformated_path = circle_points + deformation_points
                else:
                    deformated_path = circle_points
                deformated_closed_path = torch.cat((deformated_path, deformated_path[:2]), dim=-1)
                path["points"] = deformated_closed_path.view(-1, 2)

                inc = self.fill_color
                if self.WITH_OPACITY:
                    path["fill_color"] = shape_params[index: index + inc]
                else:
                    path["fill_color"] = torch.cat((shape_params[index: index + inc],
                                                    torch.tensor([1.0]).to(radius.device)), dim=-1)
                index += inc

                path["stroke_width"] = torch.tensor(0.0).to(noise.device)
                path["stroke_color"] = path["fill_color"]
                paths.append(path)

            if return_diffvg_svg_params:
                svg_params = to_diffvg_svg_params(paths=paths,
                                                  background_color=background_color,
                                                  canvas_size=self.canvas_size_)
                result_svg_params.append(svg_params)
            else:
                image = action(
                    paths=paths,
                    background_color=background_color,
                    segment_count=self.path_segment_count_,
                    canvas_size=self.canvas_size_
                )
                result.append(image)

        if not return_psvg:
            result = torch.stack(result)
            batch_size = audio_embedding.shape[0]
            result_channels = 3  # RGB
            assert result.shape == (batch_size, result_channels, self.canvas_size_, self.canvas_size_)

        if return_diffvg_svg_params:
            return result_svg_params

        return result
