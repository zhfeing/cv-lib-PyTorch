import math
from typing import Callable, List, Type

import torch
import torch.nn as nn

from torchvision.models.resnet import conv1x1, conv3x3


__all__ = [
    "WideResNet",
    "wrn_40_2",
    "wrn_40_1",
    "wrn_16_2",
    "wrn_16_1"
]


class BasicBlock(nn.Module):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        stride: int,
        drop_rate: float = 0,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        self.bn1 = norm_layer(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_planes, out_planes, stride=stride)
        self.bn2 = norm_layer(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)

        self.dropout = nn.Identity()
        if drop_rate > 0:
            self.dropout = nn.Dropout(drop_rate)

        self.equal_in_out = (in_planes == out_planes)
        self.conv_shortcut = nn.Identity()
        if not self.equal_in_out:
            self.conv_shortcut = conv1x1(in_planes, out_planes, stride=stride)

    def forward_equal(self, x: torch.Tensor) -> torch.Tensor:
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)

        out += x
        return out

    def forward_nonequal(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn1(x)
        x = self.relu1(x)

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out += self.conv_shortcut(x)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.equal_in_out:
            fn = self.forward_equal
        else:
            fn = self.forward_nonequal
        return fn(x)


class NetworkBlock(nn.Module):
    def __init__(
        self,
        nb_layers: int,
        in_planes: int,
        out_planes: int,
        block: Type[BasicBlock],
        stride,
        drop_rate: float = 0,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        layers: List[BasicBlock] = []
        layers.append(block(
            in_planes=in_planes, out_planes=out_planes, stride=stride,
            drop_rate=drop_rate, norm_layer=norm_layer
        ))
        for _ in range(1, nb_layers):
            layers.append(block(
                in_planes=out_planes, out_planes=out_planes, stride=1,
                drop_rate=drop_rate, norm_layer=norm_layer
            ))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class WideResNet(nn.Module):
    def __init__(
        self,
        block: Type[BasicBlock],
        depth: int,
        num_classes: int = 10,
        in_channels: int = 3,
        widen_factor: int = 1,
        drop_rate: float = 0,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        self._norm_layer = norm_layer

        n_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6

        # 1st conv before any network block
        self.conv1 = conv3x3(in_channels, n_channels[0])
        self.block1 = NetworkBlock(n, n_channels[0], n_channels[1], block, 1, drop_rate)
        self.block2 = NetworkBlock(n, n_channels[1], n_channels[2], block, 2, drop_rate)
        self.block3 = NetworkBlock(n, n_channels[2], n_channels[3], block, 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = norm_layer(n_channels[3])
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_channels[3], num_classes)
        self.n_channels = n_channels[3]

        self._init_parameters()

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                nn.init.normal_(m.weight, 0, math.sqrt(2. / n))
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)

        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)

        self.bn1(out)
        out = self.relu(out)

        out = self.avgpool(out)
        x = torch.flatten(x, 1)
        out = self.fc(out)

        return out


def wrn_40_2(**kwargs):
    return WideResNet(BasicBlock, depth=40, widen_factor=2, **kwargs)


def wrn_40_1(**kwargs):
    return WideResNet(BasicBlock, depth=40, widen_factor=1, **kwargs)


def wrn_16_2(**kwargs):
    return WideResNet(BasicBlock, depth=16, widen_factor=2, **kwargs)


def wrn_16_1(**kwargs):
    return WideResNet(BasicBlock, depth=16, widen_factor=1, **kwargs)

