from typing import List, Optional, Union

import torch

from ..emotions import Emotion
from ..represent import as_diffvg_render, as_SVGCont, as_SVGCont2


def calc_param_dim(path_count: int, path_segment_count: int):
    # For 1 path:
    #   (segment_count * 3 + 1) (x,y) points
    #   1 stroke width
    #   2 RGBA (stroke color, fill color)
    path_params = ((path_segment_count * 3 + 1) * 2 + 1 + 4 * 2) * path_count
    # For the canvas:
    #   1 RGB (background color)
    canvas_params = 3
    return path_params + canvas_params


class Generator(torch.nn.Module):
    def __init__(self, z_dim: int, audio_embedding_dim: int, has_emotions: bool, num_layers: int, canvas_size: int,
                 path_count: int, path_segment_count: int, max_stroke_width: float):
        super(Generator, self).__init__()

        param_dim = calc_param_dim(path_count=path_count, path_segment_count=path_segment_count)
        in_features = z_dim + audio_embedding_dim
        if has_emotions:
            in_features += len(Emotion)
        out_features = param_dim
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
            torch.nn.Linear(in_features=in_features, out_features=param_dim),
            torch.nn.Sigmoid()
        ]

        self.model_ = torch.nn.Sequential(*layers)
        self.canvas_size_ = canvas_size
        self.path_count_ = path_count
        self.path_segment_count_ = path_segment_count
        self.max_stroke_width_ = max_stroke_width

    def forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                emotions: Optional[torch.Tensor], return_psvg=False):
        if emotions is not None:
            inp = torch.cat((noise, audio_embedding, emotions), dim=1)
        else:
            inp = torch.cat((noise, audio_embedding), dim=1)
        print(f"inp: {inp.shape}")
        all_shape_params = self.model_(inp)
        print(f"all_shape_params: {all_shape_params.shape}")
        assert not torch.any(torch.isnan(all_shape_params))

        action = as_SVGCont2 if return_psvg else as_diffvg_render

        result = []
        for shape_params in all_shape_params:
            index = 0

            inc = 3  # RGB, no transparency for the background
            background_color = shape_params[index: index + inc]
            index += inc

            paths = []
            for _ in range(self.path_count_):
                path = {}

                inc = (self.path_segment_count_ * 3 + 1) * 2
                path["points"] = (shape_params[index: index + inc].view(-1, 2) * 2 - 0.5) * self.canvas_size_
                index += inc

                path["stroke_width"] = shape_params[index] * self.max_stroke_width_ * self.canvas_size_
                index += 1

                # Colors
                inc = 4  # RGBA
                path["stroke_color"] = shape_params[index: index + inc]
                index += inc
                path["fill_color"] = shape_params[index: index + inc]
                index += inc

                paths.append(path)

            assert len(paths) == self.path_count_
            image = action(
                paths=paths,
                background_color=background_color,
                canvas_size=self.canvas_size_
            )
            result.append(image)

        if not return_psvg:
            result = torch.stack(result)
            batch_size = audio_embedding.shape[0]
            result_channels = 3  # RGB
            assert result.shape == (batch_size, result_channels, self.canvas_size_, self.canvas_size_)

        return result
