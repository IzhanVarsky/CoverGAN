import random
from typing import List, Optional, Union

import numpy as np
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

import protosvg.protosvg_pb2 as psvg
from colorer.models.colorer import Colorer
from colorer.test_model import get_palette_predictor
from ..emotions import Emotion
from ..represent import *


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


def sample_circle(r, angles, sample_rate=10):
    pos = []
    for i in range(1, sample_rate + 1):
        # x = (torch.cos(angles * (sample_rate / i)) * r)  # + r
        # y = (torch.sin(angles * (sample_rate / i)) * r)  # + r
        x = (torch.cos(angles * (sample_rate / i)) * r) + r
        y = (torch.sin(angles * (sample_rate / i)) * r) + r
        pos.append(x)
        pos.append(y)
    return torch.stack(pos, dim=-1)


def soft_composite(**kwargs):
    layers = kwargs['layers']
    z_layers = kwargs['z_layers']
    n = len(layers)

    inv_mask = (1 - layers[0][:, 3:4, :, :])
    for i in range(1, n):
        inv_mask = inv_mask * (1 - layers[i][:, 3:4, :, :])

    sum_alpha = layers[0][:, 3:4, :, :] * z_layers[0]
    for i in range(1, n):
        sum_alpha = sum_alpha + layers[i][:, 3:4, :, :] * z_layers[i]
    sum_alpha = sum_alpha + inv_mask

    inv_mask = inv_mask / sum_alpha

    rgb = layers[0][:, :3] * layers[0][:, 3:4, :, :] * z_layers[0] / sum_alpha
    for i in range(1, n):
        rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * z_layers[i] / sum_alpha
    rgb = rgb * (1 - inv_mask) + inv_mask
    return rgb


def hard_composite(**kwargs):
    layers = kwargs['layers']
    n = len(layers)
    alpha = (1 - layers[n - 1][:, 3:4, :, :])
    rgb = layers[n - 1][:, :3] * layers[n - 1][:, 3:4, :, :]
    for i in reversed(range(n - 1)):
        rgb = rgb + layers[i][:, :3] * layers[i][:, 3:4, :, :] * alpha
        alpha = (1 - layers[i][:, 3:4, :, :]) * alpha
    rgb = rgb + alpha
    return rgb


def make_tensor(x, grad=False):
    x = torch.tensor(x, dtype=torch.float32)
    x.requires_grad = grad
    return x


class MyGenerator(nn.Module):
    def __init__(self, z_dim: int, audio_embedding_dim: int, has_emotions: bool, num_layers: int, canvas_size: int,
                 path_count: int, path_segment_count: int, max_stroke_width: float):
        super(MyGenerator, self).__init__()
        self.palette_predictor = get_palette_predictor()

        param_dim = 128
        in_features = z_dim + audio_embedding_dim
        if has_emotions:
            in_features += len(Emotion)
        out_features = param_dim

        kwargs = {
            "name": "VectorVAEnLayers",
            "all_figures": 4,
            "in_channels": 3,
            "latent_dim": 128,
            "loss_fn": 'MSE',
            # "paths": 20,
            "paths": 10,
            "beta": 0,
            "radius": 10,  # 3
            "scale_factor": 1,
            "learn_sampling": False,
            "only_auxillary_training": False,
            "memory_leak_training": False,
            "other_losses_weight": 0,
            "composite_fn": 'hard',
        }
        latent_dim = kwargs["latent_dim"]
        imsize = 128
        # paths = 4
        self.latent_dim = latent_dim
        self.imsize = imsize
        self.other_losses_weight = 0
        self.reparametrize_ = False
        if 'other_losses_weight' in kwargs.keys():
            self.other_losses_weight = kwargs['other_losses_weight']
        if 'reparametrize' in kwargs.keys():
            self.reparametrize_ = kwargs['reparametrize']

        self.curves = kwargs["paths"]
        self.scale_factor = kwargs['scale_factor']
        self.learn_sampling = kwargs['learn_sampling']
        self.only_auxillary_training = kwargs['only_auxillary_training']

        self.circle_rad = kwargs['radius']
        self.number_of_points = self.curves * 3

        sample_rate = 4
        angles = torch.arange(0, self.number_of_points, dtype=torch.float32) * 6.28319 / self.number_of_points
        id = sample_circle(self.circle_rad, angles, sample_rate)
        base_control_features = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.float32)
        self.id = id[:, :]
        self.angles = angles
        self.register_buffer('base_control_features', base_control_features)
        self.deformation_range = 6.28319 / 4

        def get_computational_unit(in_channels, out_channels, unit):
            if unit == 'conv':
                return nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2, padding_mode='circular', stride=1,
                                 dilation=1)
            else:
                return nn.Linear(in_channels, out_channels)

        unit = 'conv'
        if unit == 'conv':
            self.decode_transform = lambda x: x.permute(0, 2, 1)
        else:
            self.decode_transform = lambda x: x
        num_one_hot = base_control_features.shape[1]
        fused_latent_dim = latent_dim + num_one_hot + (sample_rate * 2)
        self.decoder_input = get_computational_unit(fused_latent_dim, fused_latent_dim * 2, unit)

        self.point_predictor = nn.ModuleList([
            get_computational_unit(fused_latent_dim * 2, fused_latent_dim * 2, unit),
            get_computational_unit(fused_latent_dim * 2, fused_latent_dim * 2, unit),
            get_computational_unit(fused_latent_dim * 2, fused_latent_dim * 2, unit),
            get_computational_unit(fused_latent_dim * 2, fused_latent_dim * 2, unit),
            get_computational_unit(fused_latent_dim * 2, 2, unit),
            # nn.Sigmoid()  # bound spatial extent
        ])
        if self.learn_sampling:
            self.sample_deformation = nn.Sequential(
                get_computational_unit(latent_dim + 2 + (sample_rate * 2), latent_dim * 2, unit),
                nn.ReLU(),
                get_computational_unit(latent_dim * 2, latent_dim * 2, unit),
                nn.ReLU(),
                get_computational_unit(latent_dim * 2, 1, unit),
            )
        self.aux_network = nn.Sequential(
            get_computational_unit(latent_dim, latent_dim * 2, 'mlp'),
            nn.LeakyReLU(),
            get_computational_unit(latent_dim * 2, latent_dim * 2, 'mlp'),
            nn.LeakyReLU(),
            get_computational_unit(latent_dim * 2, latent_dim * 2, 'mlp'),
            nn.LeakyReLU(),
            get_computational_unit(latent_dim * 2, 3, 'mlp'),
        )
        self.latent_lossvpath = {}
        self.save_lossvspath = False
        if self.only_auxillary_training:
            self.save_lossvspath = True
            for name, param in self.named_parameters():
                if 'aux_network' in name:
                    print(name)
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        # self.lpips = VGGPerceptualLoss(False)

        # self.colors = [[252 / 255, 194 / 255, 27 / 255, 1], [255 / 255, 0 / 255, 0 / 255, 1],
        #                [0 / 255, 255 / 255, 0 / 255, 1], [0 / 255, 0 / 255, 255 / 255, 1], ]

        self.rnn = nn.LSTM(latent_dim, latent_dim, 2, bidirectional=True)
        self.composite_fn = hard_composite
        if kwargs['composite_fn'] == 'soft':
            print('Using Differential Compositing')
            self.composite_fn = soft_composite
        self.divide_shape = nn.Sequential(nn.ReLU())
        self.final_shape_latent = nn.Sequential(
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
            get_computational_unit(latent_dim, latent_dim, 'mlp'),
            nn.ReLU(),  # bound spatial extent
        )
        self.z_order = nn.Sequential(get_computational_unit(latent_dim, 1, 'mlp'))
        layer_id = torch.tensor([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float32)
        self.register_buffer('layer_id', layer_id)

        self.all_figures = kwargs['all_figures']
        self.colors_layer = nn.Sequential(torch.nn.Linear(in_features=in_features, out_features=128),
                                          torch.nn.BatchNorm1d(num_features=128),
                                          torch.nn.LeakyReLU(0.2),
                                          torch.nn.Linear(in_features=128, out_features=self.all_figures * 3),
                                          torch.nn.Sigmoid())
        # ===============================

        self.circle_center = 2
        self.circle_radius = 1
        self.background_color_count = 3  # RGB, no transparency for the background
        self.paths_count = path_count
        # self.paths_count = 0
        self.circles_count = 0

        # points, stroke_width, stroke_color, fill_color
        # self.points_count_for_path = (path_segment_count * 3 + 1) * 2 + 1 + 4 + 4 + \
        #                              self.circle_center + self.circle_radius
        self.points_count_for_path = (path_segment_count * 3 + 1) * 2 + 1 + 4 + 4
        self.points_count_for_circle = 11
        out_dim = self.background_color_count + \
                  self.paths_count * self.points_count_for_path + \
                  self.circles_count * self.points_count_for_circle

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

        self.model_ = torch.nn.Sequential(*layers)
        self.canvas_size_ = canvas_size
        self.path_segment_count_ = path_segment_count
        self.max_stroke_width_ = max_stroke_width

    def forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                emotions: Optional[torch.Tensor], return_psvg=False) \
            -> Union[torch.Tensor, List[psvg.ProtoSVG]]:
        forward_fun = self.im2vec_forward
        # forward_fun = self.merged_ilya_im2vec_forward
        # forward_fun = self.ilya_forward
        forward_fun = self.my_mega_forward
        return forward_fun(noise, audio_embedding, emotions, return_psvg)

    def merged_ilya_im2vec_forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                                   emotions: Optional[torch.Tensor], return_psvg=False) -> torch.Tensor:
        if emotions is not None:
            inp = torch.cat((noise, audio_embedding, emotions), dim=1)
        else:
            inp = torch.cat((noise, audio_embedding), dim=1)

        all_shape_params = self.model_(inp)
        assert not torch.any(torch.isnan(all_shape_params))

        colors = self.colors_layer(inp)
        figure_paths_and_colors = self.get_im2vec_figure_paths(all_shape_params, colors)

        result = []
        for batch_ind, shape_params in enumerate(all_shape_params):
            index = 0

            inc = 3  # RGB, no transparency for the background
            background_color = shape_params[index: index + inc]
            index += inc

            paths = []
            for _ in range(self.paths_count):
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

            for im2vec_fig in figure_paths_and_colors[batch_ind]:
                new_fig = {}
                new_fig["points"] = im2vec_fig[0]
                new_fig["stroke_width"] = torch.tensor(0)
                new_fig["stroke_color"] = im2vec_fig[1]
                new_fig["fill_color"] = im2vec_fig[1]
                paths.append(new_fig)

            image = as_diffvg_render(
                paths=paths,
                background_color=background_color,
                segment_count=self.path_segment_count_,
                canvas_size=self.canvas_size_
            )
            result.append(image)

        result = torch.stack(result)
        batch_size = audio_embedding.shape[0]
        result_channels = 3  # RGB
        assert result.shape == (batch_size, result_channels, self.canvas_size_, self.canvas_size_)

        return result

    def ilya_forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                     emotions: Optional[torch.Tensor], return_psvg=False, return_diffvg_svg_params=False) \
            -> Union[torch.Tensor, List[psvg.ProtoSVG]]:
        # -> Union[torch.Tensor, List[psvg.ProtoSVG],
        # List[int, int, List[Union[Rect, Circle, Path]], List[ShapeGroup]]]:
        def random_sample(arr: np.array, size: int = 1):
            return arr[np.random.choice(len(arr), size=size, replace=False)[0]]

        def get_color_xxx(tens2, arr: np.array, size: int = 1):
            nonlocal color_ind
            nonlocal b_idx
            # samp = random_sample(arr, size)
            samp = arr[b_idx][color_ind]
            tens = torch.from_numpy(samp).to(noise.device)
            color_ind += 1
            return torch.cat((tens, tens2), dim=0)

        NEED_TO_GENERATE_COLORS = True
        if NEED_TO_GENERATE_COLORS:
            fwd = self.palette_predictor(noise, audio_embedding, emotions).detach().cpu().numpy()
            predicted_palette = fwd.reshape(fwd.shape[0], -1, 3)
        if emotions is not None:
            inp = torch.cat((noise, audio_embedding, emotions), dim=1)
        else:
            inp = torch.cat((noise, audio_embedding), dim=1)

        all_shape_params = self.model_(inp)
        assert not torch.any(torch.isnan(all_shape_params))

        action = as_protosvg if return_psvg else as_diffvg_render

        result = []
        result_svg_params = []
        for b_idx, shape_params in enumerate(all_shape_params):
            color_ind = 0
            index = 0

            inc = self.background_color_count
            if NEED_TO_GENERATE_COLORS:
                background_color = get_color_xxx(shape_params[None, index + inc - 2], predicted_palette)
            else:
                background_color = shape_params[index: index + inc]
            index += inc

            paths = []
            for _ in range(self.paths_count):
                path = {}

                inc = (self.path_segment_count_ * 3 + 1) * 2
                path["points"] = (shape_params[index: index + inc].view(-1, 2) * 2 - 0.5) * self.canvas_size_
                index += inc

                path["stroke_width"] = shape_params[index] * self.max_stroke_width_ * self.canvas_size_
                index += 1

                # Colors
                inc = 4  # RGBA
                if NEED_TO_GENERATE_COLORS:
                    path["stroke_color"] = get_color_xxx(shape_params[None, index + inc - 2], predicted_palette)
                else:
                    path["stroke_color"] = shape_params[index: index + inc]
                index += inc
                if NEED_TO_GENERATE_COLORS:
                    path["fill_color"] = get_color_xxx(shape_params[None, index + inc - 2], predicted_palette)
                else:
                    path["fill_color"] = shape_params[index: index + inc]
                index += inc

                paths.append(path)

            for _ in range(self.circles_count):
                path = {"is_circle": True, "radius": shape_params[index] * (self.canvas_size_ / 2)}
                index += 1
                inc = 2
                path["center"] = shape_params[index: index + inc] * self.canvas_size_
                inc = 4  # RGBA
                if NEED_TO_GENERATE_COLORS:
                    path["stroke_color"] = get_color_xxx(shape_params[None, index + inc - 2], predicted_palette)
                else:
                    path["stroke_color"] = shape_params[index: index + inc]
                index += inc
                if NEED_TO_GENERATE_COLORS:
                    path["fill_color"] = get_color_xxx(shape_params[None, index + inc - 2], predicted_palette)
                else:
                    path["fill_color"] = shape_params[index: index + inc]
                index += inc
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

    def im2vec_forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                       emotions: Optional[torch.Tensor], return_psvg=False) \
            -> Union[torch.Tensor, List[psvg.ProtoSVG]]:
        if emotions is not None:
            inp = torch.cat((noise, audio_embedding, emotions), dim=1)
        else:
            inp = torch.cat((noise, audio_embedding), dim=1)

        colors = self.colors_layer(inp)

        # input = self.modelXXX(inp)

        # input: torch.Size([9, 3, 128, 128])
        # mu: torch.Size([9, 128]), log_var: torch.Size([9, 128])
        # z: torch.Size([9, 128])
        # z_rnn_input: torch.Size([4, 9, 128])
        # outputs1: torch.Size([4, 9, 256])
        # hiddens[0]: torch.Size([4, 9, 128])
        # hiddens[1]: torch.Size([4, 9, 128])
        # outputs2: torch.Size([9, 4, 128])
        # output: torch.Size([9, 3, 128, 128]), control_loss: torch.Size([])
        z = self.model_(inp)
        # print(f"z: {z.shape}")
        output, control_loss = self.decode_and_composite(z, colors, verbose=False, return_overlap_loss=True)
        # print(f"output: {output.shape}, control_loss: {control_loss.shape}")
        # output = output.permute(0, 2, 3, 1)
        return output
        # return [output, input, mu, log_var, control_loss]

    def get_im2vec_figure_paths(self, z: Tensor, colors):
        bs = z.shape[0]
        figures = [[] for _ in range(bs)]

        n = self.all_figures
        z_rnn_input = z[None, :, :].repeat(n, 1, 1)  # [len, batch size, emb dim]
        outputs, hiddens = self.rnn(z_rnn_input)
        outputs = outputs.permute(1, 0, 2)  # [batch size, len, emb dim]
        outputs = outputs[:, :, :self.latent_dim] + outputs[:, :, self.latent_dim:]
        for i in range(n):
            shape_output = self.divide_shape(outputs[:, i, :])
            shape_latent = self.final_shape_latent(shape_output)
            all_points = self.decode(shape_latent)  # , point_predictor=self.point_predictor[i])
            all_points = all_points[:, :self.number_of_points, ]

            render_size = self.imsize
            if render_size == 0:
                print(f"Found size: {0}. All_points: {all_points}. Colors: {colors}. Num: {i}.")
                render_size = 128
            all_points = all_points * render_size
            num_ctrl_pts = torch.zeros(self.curves, dtype=torch.int32).to(all_points.device) + 2
            for k in range(bs):
                color = colors[k][i * 3:i * 3 + 3]
                color = torch.cat((color, torch.tensor([1]).to(all_points.device)), 0)
                # Get point parameters from network
                points = all_points[k].contiguous()  # [self.sort_idx[k]] # .cpu()

                # num_control_points=num_ctrl_pts, points=points, is_closed=True
                figures[k].append((points, color, num_ctrl_pts, True))
        return figures

    def decode_and_composite(self, z: Tensor, colors, return_overlap_loss=False, **kwargs):
        layers = []
        n = self.all_figures
        loss = 0
        z_rnn_input = z[None, :, :].repeat(n, 1, 1)  # [len, batch size, emb dim]
        # print(f"z_rnn_input: {z_rnn_input.shape}")
        outputs, hiddens = self.rnn(z_rnn_input)
        # print(f"outputs1: {outputs.shape}")
        # print(f"hiddens[0]: {hiddens[0].shape}")
        # print(f"hiddens[1]: {hiddens[1].shape}")
        outputs = outputs.permute(1, 0, 2)  # [batch size, len, emb dim]
        outputs = outputs[:, :, :self.latent_dim] + outputs[:, :, self.latent_dim:]
        # print(f"outputs2: {outputs.shape}")
        z_layers = []
        for i in range(n):
            shape_output = self.divide_shape(outputs[:, i, :])
            shape_latent = self.final_shape_latent(shape_output)
            all_points = self.decode(shape_latent)  # , point_predictor=self.point_predictor[i])
            all_points = all_points[:, :self.number_of_points, ]
            # print(torch.isfinite(all_points).all())
            # import pdb; pdb.set_trace()
            layer = self.raster(all_points, colors, i, verbose=kwargs['verbose'], white_background=False)
            z_pred = self.z_order(shape_output)
            layers.append(layer)
            z_layers.append(torch.exp(z_pred[:, :, None, None]))
            if return_overlap_loss:
                loss += self.control_polygon_distance(all_points)

        output = self.composite_fn(layers=layers, z_layers=z_layers)
        if return_overlap_loss:
            #     overlap_alpha = layers[1][:, 3:4, :, :] + layers[2][:, 3:4, :, :]
            #     loss = F.relu(overlap_alpha - 1).mean()
            return output, loss
        return output

    def raster(self, all_points, colors, num, verbose=False, white_background=True):
        # print('1:', process.memory_info().rss*1e-6)
        render_size = self.imsize
        if render_size <= 0:
            print(f"Found size: {0}. All_points: {all_points}. Colors: {colors}. Num: {num}.")
            render_size = 128
        bs = all_points.shape[0]
        if verbose:
            render_size = render_size * 2
        outputs = []
        all_points = all_points * render_size
        num_ctrl_pts = torch.zeros(self.curves, dtype=torch.int32).to(all_points.device) + 2
        for k in range(bs):
            color = colors[k][num * 3:num * 3 + 3]
            color = torch.cat((color, torch.tensor([1]).to(all_points.device)), 0)
            # Get point parameters from network
            render = pydiffvg.RenderFunction.apply
            shapes = []
            shape_groups = []
            points = all_points[k].contiguous()  # [self.sort_idx[k]] # .cpu()

            if verbose:
                np.random.seed(0)
                colors = np.random.rand(self.curves, 4)
                high = np.array((0.565, 0.392, 0.173, 1))
                low = np.array((0.094, 0.310, 0.635, 1))
                diff = (high - low) / self.curves
                colors[:, 3] = 1
                for i in range(self.curves):
                    scale = diff * i
                    color = low + scale
                    color[3] = 1
                    color = torch.tensor(color)
                    num_ctrl_pts = torch.zeros(1, dtype=torch.int32) + 2
                    if i * 3 + 4 > self.curves * 3:
                        curve_points = torch.stack([points[i * 3], points[i * 3 + 1], points[i * 3 + 2], points[0]])
                    else:
                        curve_points = points[i * 3:i * 3 + 4]
                    path = pydiffvg.Path(
                        num_control_points=num_ctrl_pts, points=curve_points,
                        is_closed=False, stroke_width=torch.tensor(4))
                    path_group = pydiffvg.ShapeGroup(
                        shape_ids=torch.tensor([i]),
                        fill_color=None,
                        stroke_color=color)
                    shapes.append(path)
                    shape_groups.append(path_group)
            else:
                path = pydiffvg.Path(
                    num_control_points=num_ctrl_pts, points=points,
                    is_closed=True)

                shapes.append(path)
                path_group = pydiffvg.ShapeGroup(
                    shape_ids=torch.tensor([len(shapes) - 1]),
                    fill_color=color,
                    stroke_color=color)
                shape_groups.append(path_group)
            scene_args = pydiffvg.RenderFunction.serialize_scene(render_size, render_size, shapes, shape_groups)
            if scene_args[0] != 128 or scene_args[1] != 128:
                print(scene_args[0], scene_args[1])
            out = render(render_size,  # width
                         render_size,  # height
                         3,  # num_samples_x
                         3,  # num_samples_y
                         102,  # seed
                         None,
                         *scene_args)
            out = out.permute(2, 0, 1).view(4, render_size, render_size)  # [:3]#.mean(0, keepdim=True)
            outputs.append(out)
        output = torch.stack(outputs).to(all_points.device)

        # map to [-1, 1]
        if white_background:
            alpha = output[:, 3:4, :, :]
            output_white_bg = output[:, :3, :, :] * alpha + (1 - alpha)
            output = torch.cat([output_white_bg, alpha], dim=1)
        del num_ctrl_pts, color
        return output

    def control_polygon_distance(self, all_points):
        def distance(vec1, vec2):
            return ((vec1 - vec2) ** 2).mean()

        loss = 0
        for idx in range(self.number_of_points):
            c_0 = all_points[:, idx - 1, :]
            c_1 = all_points[:, idx, :]
            loss = loss + distance(c_0, c_1)
        return loss

    def decode(self, z: Tensor, point_predictor=None) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        if point_predictor == None:
            point_predictor = self.point_predictor
        self.id = self.id.to(z.device)

        bs = z.shape[0]
        z = z[:, None, :].repeat([1, self.curves * 3, 1])
        base_control_features = self.base_control_features[None, :, :].repeat(bs, self.curves, 1)
        z_base = torch.cat([z, base_control_features], dim=-1)
        if self.learn_sampling:
            self.angles = self.angles.to(z.device)
            angles = self.angles[None, :, None].repeat(bs, 1, 1)
            x = torch.cos(angles)  # + r
            y = torch.sin(angles)  # + r
            z_angles = torch.cat([z_base, x, y], dim=-1)

            angles_delta = self.sample_deformation(self.decode_transform(z_angles))
            angles_delta = F.tanh(angles_delta / 50) * self.deformation_range
            angles_delta = self.decode_transform(angles_delta)

            new_angles = angles + angles_delta
            x = (torch.cos(new_angles) * self.circle_rad)  # + r
            y = (torch.sin(new_angles) * self.circle_rad)  # + r
            z = torch.cat([z_base, x, y], dim=-1)
        else:
            id = self.id[None, :, :].repeat(bs, 1, 1)
            z = torch.cat([z_base, id], dim=-1)

        all_points = self.decoder_input(self.decode_transform(z))
        for compute_block in point_predictor:
            all_points = F.relu(all_points)
            # all_points = torch.cat([z_base_transform, all_points], dim=1)
            all_points = compute_block(all_points)
        all_points = self.decode_transform(F.sigmoid(all_points / self.scale_factor))
        return all_points

    def create_circle_points(self, center, radius):
        # center = (200, 350)
        # radius = 50
        # <path fill="gray" d="M 200,300
        #              C 150,300 150,350 150,350
        #              C 150,350 150,400 200,400
        #              C 250,400 250,350 250,350
        #              C 250,300 200,300 200,300"/>
        #     <circle cx="200" cy="300" r="5" fill="red"/>
        #     <circle cx="150" cy="350" r="5" fill="yellow"/>
        #     <circle cx="200" cy="400" r="5" fill="blue"/>
        #     <circle cx="250" cy="350" r="5" fill="pink"/>
        #     <circle cx="200" cy="350" r="5" fill="purple"/>
        zero_rad = torch.cat((torch.tensor([0]).to(radius.device), radius), dim=0)
        rad_zero = torch.cat((radius, torch.tensor([0]).to(radius.device)), dim=0)
        p1 = center - zero_rad
        p2 = p1 - rad_zero
        p3 = p2 + zero_rad
        p4 = p3
        p5 = p4
        p6 = p5 + zero_rad
        p7 = p6 + rad_zero
        p8 = p7 + rad_zero
        p9 = p8 - zero_rad
        p10 = p9
        p11 = p10 - zero_rad
        p12 = p11 - rad_zero
        p13 = p12
        return torch.cat((p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13), dim=0)

    def my_mega_forward(self, noise: torch.Tensor, audio_embedding: torch.Tensor,
                        emotions: Optional[torch.Tensor], return_psvg=False, return_diffvg_svg_params=False) \
            -> Union[torch.Tensor, List[psvg.ProtoSVG]]:
        def get_color_xxx(tens2, arr: np.array, need_to_cat=True):
            nonlocal color_ind
            nonlocal b_idx
            # samp = random_sample(arr, size)
            samp = arr[b_idx][color_ind]
            tens = torch.from_numpy(samp).to(noise.device)
            color_ind += 1
            if need_to_cat:
                return torch.cat((tens, tens2), dim=0)
            return tens

        NEED_TO_GENERATE_COLORS = True
        if NEED_TO_GENERATE_COLORS:
            fwd = self.palette_predictor(noise, audio_embedding, emotions).detach().cpu().numpy()
            predicted_palette = fwd.reshape(fwd.shape[0], -1, 3)
        if emotions is not None:
            inp = torch.cat((noise, audio_embedding, emotions), dim=1)
        else:
            inp = torch.cat((noise, audio_embedding), dim=1)

        all_shape_params = self.model_(inp)
        assert not torch.any(torch.isnan(all_shape_params))

        action = as_protosvg if return_psvg else as_diffvg_render

        result = []
        result_svg_params = []
        for b_idx, shape_params in enumerate(all_shape_params):
            color_ind = 0
            index = 0

            inc = self.background_color_count
            if NEED_TO_GENERATE_COLORS:
                background_color = get_color_xxx(None, predicted_palette, False)
            else:
                background_color = shape_params[index: index + inc]
                index += inc

            paths = []
            for _ in range(self.paths_count):
                path = {}

                inc = 2
                center_point = shape_params[index: index + inc] * self.canvas_size_
                index += inc
                inc = 1
                radius = shape_params[index: index + inc] * self.canvas_size_ * 0.5
                index += inc
                circle_points = self.create_circle_points(center_point, radius)

                inc = (self.path_segment_count_ * 3 + 1) * 2
                deformation_points = shape_params[index: index + inc] * self.canvas_size_ * 0.1
                path["points"] = (circle_points + deformation_points).view(-1, 2)

                index += inc

                path["stroke_width"] = shape_params[index] * self.max_stroke_width_ * self.canvas_size_
                index += 1

                # Colors
                inc = 4  # RGBA
                if NEED_TO_GENERATE_COLORS:
                    path["stroke_color"] = get_color_xxx(shape_params[None, index + inc - 2], predicted_palette)
                else:
                    path["stroke_color"] = shape_params[index: index + inc]
                    index += inc
                if NEED_TO_GENERATE_COLORS:
                    path["fill_color"] = get_color_xxx(shape_params[None, index + inc - 2], predicted_palette)
                else:
                    path["fill_color"] = shape_params[index: index + inc]
                    index += inc
                paths.append(path)

            for _ in range(self.circles_count):
                path = {"is_circle": True, "radius": shape_params[index] * (self.canvas_size_ / 2)}
                index += 1
                inc = 2
                path["center"] = shape_params[index: index + inc] * self.canvas_size_
                inc = 4  # RGBA
                if NEED_TO_GENERATE_COLORS:
                    path["stroke_color"] = get_color_xxx(shape_params[None, index + inc - 2], predicted_palette)
                else:
                    path["stroke_color"] = shape_params[index: index + inc]
                index += inc
                if NEED_TO_GENERATE_COLORS:
                    path["fill_color"] = get_color_xxx(shape_params[None, index + inc - 2], predicted_palette)
                else:
                    path["fill_color"] = shape_params[index: index + inc]
                index += inc
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
