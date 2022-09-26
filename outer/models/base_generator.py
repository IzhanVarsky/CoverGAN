from typing import List, Optional, Union

import torch

from .cover_classes import Cover
from ..SVGContainer import SVGContainer
from ..emotions import Emotion
from ..represent import as_diffvg_render, as_SVGCont, as_SVGCont2


class BaseGenerator(torch.nn.Module):
    def fwd_cover(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                  emotions: Optional[torch.Tensor]) -> Cover:
        pass

    def fwd_svgcont(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                    emotions: Optional[torch.Tensor]) -> SVGContainer:
        pass

    def fwd_images(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                   emotions: Optional[torch.Tensor]) -> torch.Tensor:
        pass

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
