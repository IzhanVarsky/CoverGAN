import random
from typing import *

from torch import nn

from colorer.test_model import get_palette_predictor
from ..colors_tools import find_median_rgb, palette_to_triad_palette
from ..emotions import Emotion
from ..represent import *
from ..svg_tools.svg_tools import fig_classes


class MyGeneratorRandFigure(nn.Module):
    def __init__(self, z_dim: int, audio_embedding_dim: int, has_emotions: bool, num_layers: int, canvas_size: int,
                 path_count: int, path_segment_count: int, max_stroke_width: float,
                 palette_model_weights=""):
        super(MyGeneratorRandFigure, self).__init__()
        path_count = 6
        self.base_paths_count = 3
        self.max_paths_to_add = path_count - self.base_paths_count
        self.path_depth = 3
        self.radius_coef = 0.4
        self.deform_coef = 0.15
        self.max_start_path_segment_count = 5
        self.max_fig_radius_count = 2
        path_segment_count = self.max_start_path_segment_count * (2 ** (self.path_depth - 1))

        self.NEED_STROKE = False
        self.USE_PALETTE_PREDICTOR = True
        if self.USE_PALETTE_PREDICTOR:
            self.palette_predictor = get_palette_predictor(color_predictor_weights=palette_model_weights,
                                                           model_type="1")
            self.palette_predictor.eval()

        in_features = z_dim + audio_embedding_dim
        if has_emotions:
            in_features += len(Emotion)
        self.no_random_in_features = in_features - z_dim

        self.fig_center_count = 2  # (x_0, y_0)
        self.fig_angle_rotation_count = 1
        self.background_color_count = 3  # RGB, no transparency for the background
        self.all_paths_count = path_count

        self.one_path_points_count = (path_segment_count * 3) * 2
        self.all_points_count_for_path = self.fig_center_count + \
                                         self.max_fig_radius_count + \
                                         self.fig_angle_rotation_count + \
                                         self.one_path_points_count
        if self.NEED_STROKE:
            self.stroke_width_count = 1
            self.all_points_count_for_path += self.stroke_width_count
        if self.USE_PALETTE_PREDICTOR:
            # 1 = A (alpha)
            self.fill_color = 1
            self.stroke_color = 1 if self.NEED_STROKE else 0
        else:
            # 4 = RGBA
            self.fill_color = 4
            self.stroke_color = 4 if self.NEED_STROKE else 0
        self.all_points_count_for_path += self.fill_color + self.stroke_color
        self.out_dim = self.background_color_count + self.all_paths_count * self.all_points_count_for_path

        # out_features = self.out_dim
        # feature_step = (in_features - out_features) // num_layers
        # layers = []
        # for i in range(num_layers - 1):
        #     out_features = in_features - feature_step
        #     layers += [
        #         torch.nn.Linear(in_features=in_features, out_features=out_features),
        #         torch.nn.BatchNorm1d(num_features=out_features),
        #         torch.nn.LeakyReLU(0.2)
        #     ]
        #     in_features = out_features
        # layers += [
        #     torch.nn.Linear(in_features=in_features, out_features=out_dim),
        #     torch.nn.Sigmoid()
        # ]

        my_layers = [
            torch.nn.Linear(in_features=in_features, out_features=512),
            torch.nn.BatchNorm1d(num_features=512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(in_features=512, out_features=256),
            torch.nn.BatchNorm1d(num_features=256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(in_features=256, out_features=64),
            torch.nn.BatchNorm1d(num_features=64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(in_features=64, out_features=16),
            torch.nn.BatchNorm1d(num_features=16),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(in_features=16, out_features=128),
            torch.nn.BatchNorm1d(num_features=128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(in_features=128, out_features=self.out_dim),
            torch.nn.Sigmoid()
        ]
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
        self.trans = nn.TransformerEncoderLayer(d_model=self.no_random_in_features, nhead=1)
        self.model_ = torch.nn.Sequential(*my_layers)
        self.fig_add_count_model = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.no_random_in_features, out_features=self.max_paths_to_add + 1),
        )
        # self.transformer_block = TransformerBlock(1, 2, False)
        # self.attn_decoder = AttnDecoderRNN(in_features, self.out_dim)
        # self.decoder_hidden = self.attn_decoder.initHidden()
        self.sigmoid = torch.nn.Sigmoid()
        # self.rnn = nn.LSTM(in_features, self.out_dim, 2, bidirectional=True)
        self.canvas_size_ = canvas_size
        self.path_segment_count_ = path_segment_count
        self.max_stroke_width_ = max_stroke_width

    def forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                emotions: Optional[torch.Tensor], return_psvg=False, return_diffvg_svg_params=False,
                use_triad_coloring=False, palette_generator=None):
        forward_fun = self.my_mega_forward
        return forward_fun(noise, audio_embedding, emotions, return_psvg, return_diffvg_svg_params, use_triad_coloring)

    def my_mega_forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                        emotions: Optional[torch.Tensor], return_psvg=False, return_diffvg_svg_params=False,
                        use_triad_coloring=False):
        def get_color_xxx(tens2, arr: np.array):
            nonlocal color_ind
            nonlocal b_idx
            # samp = random_sample(arr, size)
            samp = arr[b_idx][color_ind % len(arr[0])]
            tens = torch.from_numpy(samp).to(noise.device)
            color_ind += 1
            if tens2 is not None:
                return torch.cat((tens, tens2), dim=0)
            return tens

        if self.USE_PALETTE_PREDICTOR:
            fwd = self.palette_predictor(noise[:, :32], audio_embedding, emotions).detach().cpu().numpy()
            predicted_palette = fwd.reshape(fwd.shape[0], -1, 3)
            if use_triad_coloring:
                predicted_palette = palette_to_triad_palette(predicted_palette)

        if emotions is not None:
            no_random_inp = torch.cat((audio_embedding, emotions), dim=1)
        else:
            no_random_inp = audio_embedding

        no_random_inp = self.trans(torch.unsqueeze(no_random_inp, 0))[0]
        vals = self.fig_add_count_model(no_random_inp).argmax(dim=1)
        inp = torch.cat((noise, no_random_inp), dim=1)
        all_shape_params = self.model_(inp)
        assert not torch.any(torch.isnan(all_shape_params))

        action = as_SVGCont2 if return_psvg else as_diffvg_render

        result = []
        result_svg_params = []
        for b_idx, shape_params in enumerate(all_shape_params):
            color_ind = 0
            index = 0

            inc = self.background_color_count
            if self.USE_PALETTE_PREDICTOR:
                background_color = get_color_xxx(None, predicted_palette)
            else:
                background_color = shape_params[index: index + inc]
                index += inc

            paths = []
            # cur_fig_cnt = self.all_paths_count
            cur_fig_cnt = self.base_paths_count + vals[b_idx]
            for _ in range(cur_fig_cnt):
                path = {}

                inc = self.fig_center_count
                center_point = shape_params[index: index + inc] * self.canvas_size_
                index += inc

                inc = self.max_fig_radius_count
                radius = shape_params[index: index + inc] * self.canvas_size_ * self.radius_coef
                index += inc

                inc = self.fig_angle_rotation_count
                angle = shape_params[index: index + inc] * 2 * math.pi
                index += inc

                inc = self.one_path_points_count
                deformation_points = (shape_params[index: index + inc] - 0.5) * 2 * radius.mean()
                deformation_points *= self.deform_coef
                index += inc

                rand_class = random.randint(0, len(fig_classes) - 1)
                # rand_class = 3
                rand_class_data = fig_classes[rand_class]
                init_fig_func = rand_class_data.base_function
                start_segment_count = rand_class_data.start_fig_segment_count
                radius = radius[0: rand_class_data.radius_count]
                deformation_points = deformation_points[:start_segment_count * 2 * 3 * (2 ** (self.path_depth - 1))]

                deformated_path = init_fig_func(center_point, radius, angle, self.path_depth, deformation_points)
                deformated_closed_path = torch.cat((deformated_path, deformated_path[:2]), dim=-1)
                path["points"] = deformated_closed_path.view(-1, 2)

                inc = self.fill_color
                if self.USE_PALETTE_PREDICTOR:
                    path["fill_color"] = get_color_xxx(shape_params[index: index + inc], predicted_palette)
                else:
                    path["fill_color"] = shape_params[index: index + inc]
                index += inc

                if self.NEED_STROKE:
                    # assert self.stroke_width_count == 1
                    path["stroke_width"] = shape_params[index] * self.max_stroke_width_ * self.canvas_size_
                    index += self.stroke_width_count

                    inc = self.stroke_color
                    if self.USE_PALETTE_PREDICTOR:
                        path["stroke_color"] = get_color_xxx(shape_params[index: index + inc], predicted_palette)
                    else:
                        path["stroke_color"] = shape_params[index: index + inc]
                    index += inc
                else:
                    path["stroke_width"] = torch.tensor(0.0).to(noise.device)
                    path["stroke_color"] = path["fill_color"]
                paths.append(path)

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
