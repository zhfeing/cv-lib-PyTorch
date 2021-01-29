from torch import Tensor
import torch
import torch.nn as nn


__all__ = [
    "conv3x3",
    "conv1x1",
    "conv3x3_pad"
]


def conv1x1(in_planes, out_planes, stride=1, bias=False):
    """1x1 convolution"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=1,
        stride=stride,
        bias=bias
    )


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=bias,
        dilation=dilation
    )


def conv3x3_pad(in_planes, out_planes, padding=0, stride=1, groups=1, dilation=1, bias=False):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=bias,
        dilation=dilation
    )


class L2Norm(nn.Module):
    def __init__(self, n_channels: int, scale: float):
        super().__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x: Tensor) -> Tensor:
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out

