import math
from typing import Optional, Union

import torch

from ..emotions import Emotion


class RasterGenerator(torch.nn.Module):
    def __init__(self, z_dim: int, audio_embedding_dim: int, has_emotions: bool, num_layers: int, canvas_size: int,
                 path_count: int, path_segment_count: int, max_stroke_width: float):
        super(RasterGenerator, self).__init__()

        in_features = z_dim + audio_embedding_dim
        if has_emotions:
            in_features += len(Emotion)

        self.start_img_dim_ = 4

        self.prep_ = torch.nn.Linear(in_features=in_features, out_features=self.start_img_dim_ ** 2)

        layers = []
        mult_iterations = int(math.log(canvas_size, 2) - math.log(self.start_img_dim_, 2))
        width = 8
        in_channels = 1
        out_channels = width * 2 ** (mult_iterations - 1)
        for _ in range(mult_iterations):
            layers += [
                torch.nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=4, padding=1, stride=2),
                torch.nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
                torch.nn.BatchNorm2d(num_features=out_channels),
                torch.nn.LeakyReLU(0.2, inplace=True),
            ]
            in_channels = out_channels
            out_channels //= 2
        out_channels *= 2
        assert out_channels == width
        layers += [
            torch.nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(num_features=width),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(in_channels=width, out_channels=width, kernel_size=3, padding=1),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Conv2d(in_channels=width, out_channels=3, kernel_size=1),  # RGB
            torch.nn.Tanh(),
        ]
        self.scaler_ = torch.nn.Sequential(*layers)
        self.canvas_size_ = canvas_size

    def forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                emotions: Optional[torch.Tensor], return_psvg=False) -> Union[torch.Tensor]:
        assert not return_psvg

        if emotions is not None:
            inp = torch.cat((noise, audio_embedding, emotions), dim=1)
        else:
            inp = torch.cat((noise, audio_embedding), dim=1)

        batch_size = audio_embedding.shape[0]
        start = self.prep_(inp).view(batch_size, 1, self.start_img_dim_, self.start_img_dim_)
        result = self.scaler_(start)
        result = (result + 1) / 2

        result_channels = 3  # RGB
        assert result.shape == (batch_size, result_channels, self.canvas_size_, self.canvas_size_)

        return result
