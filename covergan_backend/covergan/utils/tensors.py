from typing import Union

from torch import Tensor
from torch.nn.utils.rnn import PackedSequence


def get_device(t: Union[Tensor, PackedSequence]):
    if isinstance(t, PackedSequence):
        return t.data.device
    else:
        return t.device
