import collections.abc
from itertools import repeat
from typing import Callable

import torch
import torch.nn as nn


__all__ = [
    "to_1tuple", "to_2tuple", "to_3tuple", "to_4tuple",
    "get_activation_fn"
]


# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple


def get_activation_fn(activation_name: str) -> Callable[[torch.Tensor], torch.Tensor]:
    __SUPPORTED_ACTIVATION__ = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "glu": nn.GLU
    }
    return __SUPPORTED_ACTIVATION__[activation_name]()


