"""
Large resnet models for cifar datasets, same as crd repo-dist
"""

from typing import Callable, List, Union, Type

import torch
import torch.nn as nn
from torchvision.models.resnet import conv1x1, conv3x3

from .resnet import BasicBlock, Bottleneck


class BasicBlock_CL(BasicBlock):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ):
        if stride != 1 or inplanes != self.expansion * planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, self.expansion * planes, stride),
                nn.BatchNorm2d(self.expansion * planes)
            )
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            norm_layer=norm_layer
        )


class Bottleneck_CL(Bottleneck):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ):
        if stride != 1 or inplanes != self.expansion * planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, self.expansion * planes, stride),
                nn.BatchNorm2d(self.expansion * planes)
            )
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            norm_layer=norm_layer
        )


class ResNet_CL(nn.Module):
    """
    Resnet for cifar dataset (large version).

    @ Different from PyTorch version `in ()`:
        1. First conv layer has kernel size of 3 (7) and stride 1 (2)
        2. Using non-inplace relu for feature extracting
    """
    def __init__(
        self,
        block: Type[Union[BasicBlock_CL, Bottleneck_CL]],
        num_blocks: List[int],
        num_classes=10,
        in_channels: int = 3,
        zero_init_residual: bool = False,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ):
        super().__init__()
        self.inplanes = 64
        self._norm_layer = norm_layer

        self.conv1 = conv3x3(in_channels, self.inplanes)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

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
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock_CL, Bottleneck_CL]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ):
        norm_layer = self._norm_layer
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for i in range(blocks):
            stride = strides[i]
            layers.append(block(
                inplanes=self.inplanes, planes=planes,
                stride=stride, norm_layer=norm_layer
            ))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


def resnet18_cl(**kwargs):
    return ResNet_CL(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34_cl(**kwargs):
    return ResNet_CL(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50_cl(**kwargs):
    return ResNet_CL(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101_cl(**kwargs):
    return ResNet_CL(Bottleneck, [3, 4, 23, 3], **kwargs)


def resnet152_cl(**kwargs):
    return ResNet_CL(Bottleneck, [3, 8, 36, 3], **kwargs)


MODEL_DICT = {
    # large ResNet for cifar
    "ResNet18_cl": resnet18_cl,
    "ResNet34_cl": resnet34_cl,
    "ResNet50_cl": resnet50_cl,
    "ResNet101_cl": resnet101_cl,
    "ResNet152_cl": resnet152_cl,
}
