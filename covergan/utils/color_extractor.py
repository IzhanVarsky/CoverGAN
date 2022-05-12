from typing import Optional, Tuple

import numpy as np
import torch

from scipy.cluster.vq import kmeans, vq


def extract_primary_color(img: torch.Tensor, count: int) -> Optional[Tuple[int, int, int]]:
    # Requires HWC RGBA tensors
    shape = img.shape
    img = torch.reshape(img, (shape[0] * shape[1], shape[2])).numpy()

    codes, dist_ = kmeans(img, k_or_guess=count)
    img = img[:, :3]
    codes = codes[codes[:, 3] > 0.5][:, :3]  # Filter out blank background (by alpha channel)

    if not len(codes):
        return None

    vecs, dist_ = vq(img, codes)
    counts, bin_edges_ = np.histogram(vecs, bins=len(codes))  # Count occurrences

    idx = np.argmax(counts)  # Find most frequent

    result = np.round(codes[idx] * 255).astype(int)  # [0, 1] -> [0, 255]

    return tuple(map(int, result))
