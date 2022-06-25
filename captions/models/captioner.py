import torch

from utils.edges import detect_edges


def make_linear_layer(in_features: int, out_features: int, num_layers: int):
    layers = []
    out_features_final = out_features
    for i in range(num_layers - 1):
        out_features = in_features // 4
        layers += [
            torch.nn.Linear(in_features=in_features, out_features=out_features),
            torch.nn.LeakyReLU(),
        ]
        if i != num_layers - 2:
            layers.append(torch.nn.Dropout2d(0.2))
        in_features = out_features
    layers += [
        torch.nn.Linear(in_features=in_features, out_features=out_features_final),
        torch.nn.Sigmoid()
    ]
    return torch.nn.Sequential(*layers)


def make_conv(in_channels: int, out_channels: int, num_layers: int, canvas_size: int) -> (torch.nn.Module, int, int):
    layers = []
    ds_size = canvas_size

    conv_kernel_size = 3
    conv_stride = 2
    conv_padding = 1

    for i in range(num_layers):
        layers += [
            torch.nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding
            ),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(
                in_channels=out_channels, out_channels=out_channels,
                kernel_size=conv_kernel_size,
                stride=conv_stride,
                padding=conv_padding
            ),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.Dropout2d(0.2)
        ]

        in_channels = out_channels
        out_channels *= 2
        # 2 convolutions
        for _ in range(2):
            ds_size = (ds_size + 2 * conv_padding - conv_kernel_size) // conv_stride + 1
        conv_kernel_size = min(conv_kernel_size + 2, 5)
        conv_stride = max(conv_stride - 1, 2)

    out_channels //= 2
    # The height and width of downsampled image
    assert ds_size > 0
    img_dim = out_channels * (ds_size ** 2)

    return torch.nn.Sequential(*layers), img_dim


class Captioner(torch.nn.Module):

    def __init__(self, canvas_size: int, num_conv_layers: int, num_linear_layers: int):
        super(Captioner, self).__init__()

        in_channels = 4  # RGB + grayscale edges
        out_channels = in_channels * 16

        self.conv_, img_dim = make_conv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_conv_layers,
            canvas_size=canvas_size
        )

        self.pos_predictor_ = make_linear_layer(
            in_features=img_dim,
            out_features=2 * 4,  # x, y, width, height
            num_layers=num_linear_layers
        )

        self.color_predictor_ = make_linear_layer(
            in_features=img_dim,
            out_features=2 * 3,  # 2 RGB colors
            num_layers=num_linear_layers
        )

    def forward(self, img: torch.Tensor, edges: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        if edges is None:
            edges = torch.stack([detect_edges(x) for x in img]).to(img.device)
        edges = edges.unsqueeze(dim=1)
        inp = torch.cat((img, edges), dim=1)
        output = self.conv_(inp)
        output = output.reshape(output.shape[0], -1)  # Flatten elements in the batch

        pos_pred = self.pos_predictor_(output)
        color_pred = self.color_predictor_(output)
        assert not torch.any(torch.isnan(pos_pred))
        assert not torch.any(torch.isnan(color_pred))

        return pos_pred, color_pred
