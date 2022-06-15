from torch import nn
from typing import *
from colorer.test_model import get_palette_predictor
from ..emotions import Emotion
from ..represent import *
from .attn_decoder_rnn import AttnDecoderRNN
from .former.modules import TransformerBlock
from ..svg_tools.svg_tools import create_circle_control_points


class MyGeneratorCircled(nn.Module):
    def __init__(self, z_dim: int, audio_embedding_dim: int, has_emotions: bool, num_layers: int, canvas_size: int,
                 path_count: int, path_segment_count: int, max_stroke_width: float):
        super(MyGeneratorCircled, self).__init__()
        path_count = 3
        path_segment_count = 20

        self.NEED_STROKE = False
        self.USE_PALETTE_PREDICTOR = True
        if self.USE_PALETTE_PREDICTOR:
            self.palette_predictor = get_palette_predictor()

        in_features = z_dim + audio_embedding_dim
        if has_emotions:
            in_features += len(Emotion)

        self.circle_center_count = 2  # (x_0, y_0)
        self.circle_radius_count = 1
        self.background_color_count = 3  # RGB, no transparency for the background
        self.all_paths_count = path_count

        self.one_path_points_count = (path_segment_count * 3) * 2
        self.all_points_count_for_path = self.circle_center_count + \
                                         self.circle_radius_count + \
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

        out_features = self.out_dim
        feature_step = (in_features - out_features) // num_layers

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
        my_layers = [
            torch.nn.Linear(in_features=in_features, out_features=512),
            torch.nn.BatchNorm1d(num_features=512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(in_features=512, out_features=self.out_dim),
            torch.nn.Sigmoid()
        ]
        self.model_ = torch.nn.Sequential(*my_layers)
        self.transformer_block = TransformerBlock(1, 2, False)
        # self.attn_decoder = AttnDecoderRNN(in_features, self.out_dim)
        # self.decoder_hidden = self.attn_decoder.initHidden()
        self.sigmoid = torch.nn.Sigmoid()
        # self.rnn = nn.LSTM(in_features, self.out_dim, 2, bidirectional=True)
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
        def get_color_xxx(tens2, arr: np.array):
            nonlocal color_ind
            nonlocal b_idx
            # samp = random_sample(arr, size)
            samp = arr[b_idx][color_ind]
            tens = torch.from_numpy(samp).to(noise.device)
            color_ind += 1
            if tens2 is not None:
                return torch.cat((tens, tens2), dim=0)
            return tens

        if self.USE_PALETTE_PREDICTOR:
            fwd = self.palette_predictor(noise, audio_embedding, emotions).detach().cpu().numpy()
            predicted_palette = fwd.reshape(fwd.shape[0], -1, 3)

        if emotions is not None:
            inp = torch.cat((noise, audio_embedding, emotions), dim=1)
        else:
            inp = torch.cat((noise, audio_embedding), dim=1)

        inp = self.transformer_block(inp.view(inp.shape[0], -1, 1))
        inp = inp.view(inp.shape[0], -1)

        all_shape_params = self.model_(inp)
        # inp = inp[None, :, :].repeat(self.all_paths_count, 1, 1)  # [len, batch size, emb dim]
        # inp = inp[None, :, :].repeat(1, 1, 1)  # [len, batch size, emb dim]
        # # outputs, _ = self.rnn(inp)
        # outputs, hidden, attn_weights = self.attn_decoder(inp, self.decoder_hidden, inp)
        # self.decoder_hidden = hidden
        # outputs = outputs.permute(1, 0, 2)  # [batch size, len, emb dim]
        # outputs = outputs[:, :, :self.out_dim] + outputs[:, :, self.out_dim:]
        # all_shape_params = self.sigmoid(outputs)[:, 0]

        assert not torch.any(torch.isnan(all_shape_params))

        action = as_protosvg if return_psvg else as_diffvg_render

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
            for _ in range(self.all_paths_count):
                path = {}

                inc = self.circle_center_count
                center_point = shape_params[index: index + inc] * self.canvas_size_
                index += inc

                inc = self.circle_radius_count
                # radius = shape_params[index: index + inc] * self.canvas_size_ * 0.4
                radius = shape_params[index: index + inc] * self.canvas_size_ * 0.3
                index += inc

                circle_points = create_circle_control_points(center_point, radius, self.path_segment_count_)

                inc = self.one_path_points_count
                # deformation_points = (shape_params[index: index + inc] - 0.5) * 2 * self.canvas_size_ * 0.1
                # deformation_points = (shape_params[index: index + inc] - 0.5) * 2 * radius * 0.3
                # deformation_points = (shape_params[index: index + inc] - 0.5) * 2 * radius * 0.2
                deformation_points = (shape_params[index: index + inc] - 0.5) * 2 * radius * 0.1
                index += inc

                deformated_path = circle_points + deformation_points
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
