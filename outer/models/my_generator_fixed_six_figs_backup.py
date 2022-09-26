from typing import *

from torch import nn

from colorer.test_model import get_palette_predictor
from .generator_utils import colorize
from ..emotions import Emotion
from ..represent import *
from ..svg_tools.svg_tools import *


class MyGeneratorFixedSixFigs32(nn.Module):
    def __init__(self, z_dim: int, audio_embedding_dim: int, has_emotions: bool, num_layers: int, canvas_size: int,
                 path_count: int, path_segment_count: int, max_stroke_width: float,
                 palette_model_weights=""):
        super(MyGeneratorFixedSixFigs32, self).__init__()
        self.figs_config = [
            init_func_types_config[InitFuncType.RECT],
            init_func_types_config[InitFuncType.TRIANGLE],
            init_func_types_config[InitFuncType.RECT],
            init_func_types_config[InitFuncType.TRIANGLE],
            init_func_types_config[InitFuncType.TRIANGLE],
            init_func_types_config[InitFuncType.PENTAGON],
        ]
        path_count = len(self.figs_config)
        self.path_depth = 4
        self.radius_coef = 0.35
        self.deform_coef = 0.15

        self.USE_ATTN = False
        self.NEED_STROKE = False
        self.USE_PALETTE_PREDICTOR = True
        if self.USE_PALETTE_PREDICTOR:
            self.palette_predictor = get_palette_predictor(color_predictor_weights=palette_model_weights)
            self.palette_predictor.eval()

        in_features = z_dim + audio_embedding_dim
        if has_emotions:
            in_features += len(Emotion)
        self.no_random_in_features = in_features - z_dim

        self.fig_center_count = 2  # (x_0, y_0)
        self.fig_angle_rotation_count = 1
        self.background_color_count = 3  # RGB, no transparency for the background
        self.all_paths_count = path_count

        self.deform_points_count_for_each_fig = [
            get_all_path_segment_count(conf.start_fig_segment_count, self.path_depth) * 3 * 2
            for conf in self.figs_config
        ]
        self.all_points_count_for_each_fig = [
            self.deform_points_count_for_each_fig[ind] + conf.radius_count +
            self.fig_center_count + self.fig_angle_rotation_count
            for ind, conf in enumerate(self.figs_config)
        ]

        self.addable_count = 0
        if self.NEED_STROKE:
            self.stroke_width_count = 1
            self.addable_count += self.stroke_width_count
        if self.USE_PALETTE_PREDICTOR:
            # 1 = A (alpha)
            self.fill_color = 1
            self.stroke_color = 1 if self.NEED_STROKE else 0
        else:
            # 4 = RGBA
            self.fill_color = 4
            self.stroke_color = 4 if self.NEED_STROKE else 0
        self.addable_count += self.fill_color + self.stroke_color
        self.out_dim = self.background_color_count + sum(self.all_points_count_for_each_fig) \
                       + len(self.all_points_count_for_each_fig) * self.addable_count

        out_features = self.out_dim
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
            torch.nn.Linear(in_features=in_features, out_features=self.out_dim),
            torch.nn.Sigmoid()
        ]
        my_layers = layers

        # my_layers = [
        #     torch.nn.Linear(in_features=in_features, out_features=512),
        #     torch.nn.BatchNorm1d(num_features=512),
        #     torch.nn.LeakyReLU(0.2),
        #     torch.nn.Linear(in_features=512, out_features=256),
        #     torch.nn.BatchNorm1d(num_features=256),
        #     torch.nn.LeakyReLU(0.2),
        #     torch.nn.Linear(in_features=256, out_features=64),
        #     torch.nn.BatchNorm1d(num_features=64),
        #     torch.nn.LeakyReLU(0.2),
        #     torch.nn.Linear(in_features=64, out_features=16),
        #     torch.nn.BatchNorm1d(num_features=16),
        #     torch.nn.LeakyReLU(0.2),
        #     torch.nn.Linear(in_features=16, out_features=128),
        #     torch.nn.BatchNorm1d(num_features=128),
        #     torch.nn.LeakyReLU(0.2),
        #     torch.nn.Linear(in_features=128, out_features=self.out_dim),
        #     torch.nn.Sigmoid()
        # ]
        # my_layers = [
        #     torch.nn.Linear(in_features=in_features, out_features=512),
        #     torch.nn.BatchNorm1d(num_features=512),
        #     torch.nn.LeakyReLU(0.2),
        #     torch.nn.Linear(in_features=512, out_features=self.out_dim),
        #     torch.nn.Sigmoid()
        # ]
        # my_layers = [
        #     torch.nn.Linear(in_features=in_features, out_features=self.out_dim),
        #     torch.nn.Sigmoid()
        # ]
        if self.USE_ATTN:
            self.trans = nn.TransformerEncoderLayer(d_model=self.no_random_in_features, nhead=1)

        self.model_ = torch.nn.Sequential(*my_layers)
        # self.transformer_block = TransformerBlock(1, 2, False)
        # self.attn_decoder = AttnDecoderRNN(in_features, self.out_dim)
        # self.decoder_hidden = self.attn_decoder.initHidden()
        self.sigmoid = torch.nn.Sigmoid()
        # self.rnn = nn.LSTM(in_features, self.out_dim, 2, bidirectional=True)
        self.canvas_size_ = canvas_size
        self.max_stroke_width_ = max_stroke_width

    def forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                emotions: Optional[torch.Tensor], return_psvg=False, return_diffvg_svg_params=False,
                use_triad_coloring=False, palette_generator=None):
        forward_fun = self.my_mega_forward
        return forward_fun(noise, audio_embedding, emotions, return_psvg, return_diffvg_svg_params, use_triad_coloring)

    def my_mega_forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                        emotions: Optional[torch.Tensor], return_psvg=False, return_diffvg_svg_params=False,
                        use_triad_coloring=False):
        if self.USE_PALETTE_PREDICTOR:
            fwd = self.palette_predictor(noise[:, :32], audio_embedding, emotions)
            predicted_palette = fwd.reshape(fwd.shape[0], -1, 3)

        if emotions is not None:
            no_random_inp = torch.cat((audio_embedding, emotions), dim=1)
        else:
            no_random_inp = audio_embedding

        if self.USE_ATTN:
            no_random_inp = self.trans(torch.unsqueeze(no_random_inp, 0))[0]
        inp = torch.cat((noise, no_random_inp), dim=1)
        all_shape_params = self.model_(inp)
        assert not torch.any(torch.isnan(all_shape_params))

        action = as_SVGCont2 if return_psvg else as_diffvg_render

        result = []
        result_svg_params = []
        for b_idx, shape_params in enumerate(all_shape_params):
            index = 0

            inc = self.background_color_count
            if not self.USE_PALETTE_PREDICTOR:
                background_color = shape_params[index: index + inc]
                index += inc

            paths = []
            for fig_num, fig_config in enumerate(self.figs_config):
                path = {}

                inc = self.fig_center_count
                center_point = shape_params[index: index + inc] * self.canvas_size_
                index += inc

                inc = fig_config.radius_count
                radius = shape_params[index: index + inc] * self.canvas_size_ * self.radius_coef
                index += inc

                inc = self.fig_angle_rotation_count
                angle = shape_params[index: index + inc] * 2 * math.pi
                index += inc

                inc = self.deform_points_count_for_each_fig[fig_num]
                deformation_points = (shape_params[index: index + inc] - 0.5) * 2 * radius.mean()
                deformation_points *= self.deform_coef
                index += inc

                init_fig_func = fig_config.base_function

                deformated_path = init_fig_func(center_point, radius, angle, self.path_depth, deformation_points)
                deformated_closed_path = torch.cat((deformated_path, deformated_path[:2]), dim=-1)
                path["points"] = deformated_closed_path.view(-1, 2)

                inc = self.fill_color
                path["fill_color"] = shape_params[index: index + inc]
                index += inc

                if self.NEED_STROKE:
                    # assert self.stroke_width_count == 1
                    path["stroke_width"] = shape_params[index] * self.max_stroke_width_ * self.canvas_size_
                    index += self.stroke_width_count

                    inc = self.stroke_color
                    path["stroke_color"] = shape_params[index: index + inc]
                    index += inc
                else:
                    path["stroke_width"] = torch.tensor(0.0).to(noise.device)
                    path["stroke_color"] = path["fill_color"]
                paths.append(path)
            if self.USE_PALETTE_PREDICTOR:
                background_color = colorize(paths, colors=predicted_palette[b_idx], use_triad=use_triad_coloring,
                                            need_stroke=self.NEED_STROKE)
            if return_diffvg_svg_params:
                svg_params = to_diffvg_svg_params(paths=paths,
                                                  background_color=background_color,
                                                  canvas_size=self.canvas_size_)
                result_svg_params.append(svg_params)
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

        if return_diffvg_svg_params:
            return result_svg_params, result

        return result
