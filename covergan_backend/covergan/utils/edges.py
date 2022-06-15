import torch

from skimage.feature import canny
from kornia.color import rgb_to_grayscale


def detect_edges(img: torch.Tensor) -> torch.Tensor:
    # The input image is expected to be CxHxW
    return torch.tensor(canny(rgb_to_grayscale(img).squeeze(0).cpu().numpy()))
