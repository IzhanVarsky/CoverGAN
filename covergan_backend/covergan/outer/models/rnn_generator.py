from typing import List, Optional, Union

import torch

from ..emotions import Emotion
from ..represent import as_diffvg_render, as_protosvg
from utils.weight_init import weights_init_uniform
from protosvg.client import color_to_int
import protosvg.protosvg_pb2 as psvg


def tensor_color_to_int(t: torch.Tensor, a: float):
    t = (t * 255).round().to(int)
    r = t[0].item()
    g = t[1].item()
    b = t[2].item()
    a = round(255 * a)
    return color_to_int(r, g, b, a)


class RNNGenerator(torch.nn.Module):
    def __init__(self, z_dim: int, audio_embedding_dim: int, has_emotions: bool, num_layers: int, canvas_size: int,
                 path_count: int, path_segment_count: int, max_stroke_width: float):
        super(RNNGenerator, self).__init__()

        in_features = z_dim + audio_embedding_dim
        if has_emotions:
            in_features += len(Emotion)
        hidden_size = 256

        self.hidden_cell_predictor = torch.nn.Linear(
            in_features=z_dim + len(Emotion) if has_emotions else z_dim,
            out_features=num_layers * hidden_size * 2  # c_0 and h_0
        )

        self.lstm = torch.nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.7,
            batch_first=True
        )

        self.point_predictor = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hidden_size,
                out_features=(path_segment_count * 3 + 1) * 2
            ),  # 1 start point + 3 (x,y) points
            torch.nn.Tanh()
        )

        self.stroke_width_predictor = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=1),
            torch.nn.Sigmoid()
        )

        self.alpha_predictor = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=1),  # A
            torch.nn.Sigmoid()
        )

        self.stroke_color_predictor = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=3),  # RGB (A is above)
            torch.nn.Sigmoid()
        )

        self.fill_color_predictor = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=3),  # RGB (A is above)
            torch.nn.Sigmoid()
        )

        self.bg_color_predictor = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_size, out_features=3),  # RGB (no A channel)
            torch.nn.Sigmoid()
        )

        # Initial weights are overridden by checkpoint restoration (if any)
        weights_init_uniform(self.point_predictor, 1.0)
        weights_init_uniform(self.fill_color_predictor, 1.0)
        weights_init_uniform(self.bg_color_predictor, 1.0)

        self.hidden_size_ = hidden_size
        self.num_layers_ = num_layers
        self.canvas_size_ = canvas_size
        self.path_segment_count_ = path_segment_count
        self.max_stroke_width_ = max_stroke_width

    def forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                emotions: Optional[torch.Tensor], return_psvg=False) -> Union[torch.Tensor, List[psvg.ProtoSVG]]:
        batch_size, slice_count, audio_embedding_dim = audio_embedding.shape

        repeated_noise = noise.unsqueeze(dim=1).repeat(1, slice_count, 1)
        if emotions is not None:
            repeated_emotions = emotions.unsqueeze(dim=1).repeat(1, slice_count, 1)
            inp = torch.cat((repeated_noise, audio_embedding, repeated_emotions), dim=2)
            hidden_and_cell = self.hidden_cell_predictor(torch.tanh(torch.cat((noise, emotions), dim=1)))
        else:
            inp = torch.cat((repeated_noise, audio_embedding), dim=2)
            hidden_and_cell = self.hidden_cell_predictor(torch.tanh(noise))

        hidden = hidden_and_cell[:, :self.num_layers_ * self.hidden_size_]  # 1st half for every in batch
        hidden = hidden.view(batch_size, self.num_layers_, self.hidden_size_)
        hidden = hidden.permute(1, 0, 2).contiguous()
        cell = hidden_and_cell[:, self.num_layers_ * self.hidden_size_:]  # 2nd half for every in batch
        cell = cell.view(batch_size, self.num_layers_, self.hidden_size_)
        cell = cell.permute(1, 0, 2).contiguous()
        hidden_and_cell = (hidden, cell)

        feats, hidden_and_cell = self.lstm(inp, hidden_and_cell)

        # Reshape to make it a valid input size for the predictors
        feats = feats.reshape(batch_size * slice_count, self.hidden_size_)

        # Point coordinates are relative to the canvas
        # [-1, 1] -> [-0.5, 1.5] (Tanh, [0, 1] is the canvas, [-0.5, 0] and [1, 1.5] are off-canvas blind spots)
        all_points = self.point_predictor(feats).view(batch_size, slice_count, self.path_segment_count_ * 3 + 1, 2)
        all_points = (all_points + 0.5) * self.canvas_size_
        # Stroke widths are relative to the max stroke width, not canvas
        all_stroke_widths = self.stroke_width_predictor(feats).view(batch_size, slice_count) * self.max_stroke_width_
        all_stroke_widths = all_stroke_widths * self.canvas_size_

        # Colors
        all_alphas = self.alpha_predictor(feats).view(batch_size, slice_count)
        all_stroke_colors = self.stroke_color_predictor(feats).view(batch_size, slice_count, 3)
        all_fill_colors = self.fill_color_predictor(feats).view(batch_size, slice_count, 3)
        all_bg_colors = self.bg_color_predictor(feats).view(batch_size, slice_count, 3)[:, -1]  # One per canvas!

        action = as_protosvg if return_psvg else as_diffvg_render

        result = []
        for b in range(batch_size):
            paths = []
            for i in range(slice_count):
                alpha = all_alphas[b, i].unsqueeze(dim=0)
                path = {
                    "points": all_points[b, i],
                    "stroke_width": all_stroke_widths[b, i],
                    "stroke_color": torch.cat((all_stroke_colors[b, i], alpha)),
                    "fill_color": torch.cat((all_fill_colors[b, i], alpha))
                }
                paths.append(path)
            image = action(
                paths=paths,
                background_color=all_bg_colors[b],
                segment_count=self.path_segment_count_,
                canvas_size=self.canvas_size_
            )
            result.append(image)

        if not return_psvg:
            result = torch.stack(result)
            result_channels = 3  # RGB
            assert result.shape == (batch_size, result_channels, self.canvas_size_, self.canvas_size_)

        return result
