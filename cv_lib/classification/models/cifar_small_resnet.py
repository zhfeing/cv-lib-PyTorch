"""
Small resnet models for cifar datasets, same as crd repo-dist
"""

from typing import Callable, List, Union, Type

import torch
import torch.nn as nn
from torchvision.models.resnet import conv1x1, conv3x3

from .resnet import BasicBlock, Bottleneck


class BasicBlock_CS(BasicBlock):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = nn.Identity(),
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            norm_layer=norm_layer
        )


class Bottleneck_CS(Bottleneck):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module = nn.Identity(),
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            norm_layer=norm_layer
        )


class ResNet_CS(nn.Module):
    """
    Resnet for cifar dataset.

    @ Different from PyTorch version, noted `in ()`:
        1. First conv layer has kernel size of 3 (7) and stride 1 (2)
        2. Using non-inplace relu for feature extracting
        3. Only 3 (4) residual blocks
        4. Much smaller channel size
    """
    def __init__(
        self,
        block: Type[Union[BasicBlock_CS, Bottleneck_CS]],
        num_filters: List[int],
        depth: int,
        num_classes=10,
        in_channels: int = 3,
        zero_init_residual: bool = False,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        self._norm_layer = norm_layer
        # Model type specifies number of layers for CIFAR-10 model
        if block == BasicBlock_CS:
            assert (depth - 2) % 6 == 0, "When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202"
            n = (depth - 2) // 6
        elif block == Bottleneck_CS:
            assert (depth - 2) % 9 == 0, "When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199"
            n = (depth - 2) // 9
        else:
            raise ValueError("block shoule be Basicblock or Bottleneck")

        self.inplanes = num_filters[0]
        self.conv1 = conv3x3(in_channels, num_filters[0])
        self.bn1 = norm_layer(num_filters[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, num_filters[1], n)
        self.layer2 = self._make_layer(block, num_filters[2], n, stride=2)
        self.layer3 = self._make_layer(block, num_filters[3], n, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(num_filters[3] * block.expansion, num_classes)

        self._init_parameters(zero_init_residual)

    def _init_parameters(self, zero_init_residual: bool):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck_CS):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock_CS):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock_CS, Bottleneck_CS]],
        planes: int,
        blocks: int,
        stride: int = 1
    ):
        norm_layer = self._norm_layer
        downsample = nn.Identity()
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers: List[Union[BasicBlock_CS, Bottleneck_CS]] = []
        layers.append(block(
            inplanes=self.inplanes, planes=planes, stride=stride,
            downsample=downsample, norm_layer=norm_layer
        ))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                inplanes=self.inplanes, planes=planes,
                norm_layer=norm_layer
            ))

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)  # 32x32

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def resnet8_cs(**kwargs):
    return ResNet_CS(BasicBlock_CS, [16, 16, 32, 64], 8, **kwargs)


def resnet14_cs(**kwargs):
    return ResNet_CS(BasicBlock_CS, [16, 16, 32, 64], 14, **kwargs)


def resnet20_cs(**kwargs):
    return ResNet_CS(BasicBlock_CS, [16, 16, 32, 64], 20, **kwargs)


def resnet32_cs(**kwargs):
    return ResNet_CS(BasicBlock_CS, [16, 16, 32, 64], 32, **kwargs)


def resnet44_cs(**kwargs):
    return ResNet_CS(BasicBlock_CS, [16, 16, 32, 64], 44, **kwargs)


def resnet56_cs(**kwargs):
    return ResNet_CS(BasicBlock_CS, [16, 16, 32, 64], 56, **kwargs)


def resnet110_cs(**kwargs):
    return ResNet_CS(BasicBlock_CS, [16, 16, 32, 64], 110, **kwargs)


def resnet8x4_cs(**kwargs):
    return ResNet_CS(BasicBlock_CS, [32, 64, 128, 256], 8, **kwargs)


def resnet32x4_cs(**kwargs):
    return ResNet_CS(BasicBlock_CS, [32, 64, 128, 256], 32, **kwargs)


MODEL_DICT = {
    # small resnet for cifar
    "resnet20_cs": resnet20_cs,
    "resnet32_cs": resnet32_cs,
    "resnet56_cs": resnet56_cs,
    "resnet44_cs": resnet44_cs,
    "resnet56_cs": resnet56_cs,
    "resnet110_cs": resnet110_cs,
    "resnet8x4_cs": resnet8x4_cs,
    "resnet32x4_cs": resnet32x4_cs,
}
