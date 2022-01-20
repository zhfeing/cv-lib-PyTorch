from typing import List

from torch import Tensor
import torch.nn as nn
from torch.nn import Sequential, ModuleList, Module

from .aux_layers import conv1x1, conv3x3, conv3x3_pad
from .backbones import Backbone
from .sub_module import BoxPredictor


__all__ = [
    "SSD300_ResNet"
]


class SSD300_ResNet(Module):
    """
    Only accept 300x300 input image

    Support manual bn layer
    """
    def __init__(
        self,
        backbone: Backbone,
        out_channels: List[int] = [1024, 512, 512, 256, 256, 256],
        num_default_boxes: List[int] = [4, 6, 6, 6, 4, 4],
        n_classes: int = 81,
        norm_layer: Module = None
    ):
        super().__init__()

        assert len(out_channels) == len(num_default_boxes) == 6, "only support 6 middle features"

        self.backbone = backbone
        self.detection_header = BoxPredictor(n_classes, out_channels, num_default_boxes)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        # used for extracted features
        self._middle_channels = [256, 256, 128, 128, 128]
        self.additional_blocks = self._build_additional_features(out_channels, norm_layer)
        self._init_weights()

    def _build_additional_features(self, out_channels: List[int], norm_layer: Module) -> ModuleList:
        additional_blocks = ModuleList()
        for i, (input_size, out_c, channels) in enumerate(zip(
            out_channels[:-1],
            out_channels[1:],
            self._middle_channels
        )):
            if i < 3:
                layer = Sequential(
                    conv1x1(input_size, channels),
                    norm_layer(channels),
                    nn.ReLU(inplace=True),
                    conv3x3(channels, out_c, stride=2),
                    norm_layer(out_c),
                    nn.ReLU(inplace=True),
                )
            else:
                layer = Sequential(
                    conv1x1(input_size, channels),
                    norm_layer(channels),
                    nn.ReLU(inplace=True),
                    conv3x3_pad(channels, out_c),
                    norm_layer(out_c),
                    nn.ReLU(inplace=True),
                )
            additional_blocks.append(layer)
        return additional_blocks

    def _init_weights(self):
        layers = [*self.additional_blocks]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    def forward(self, x: Tensor):
        x = self.backbone(x)

        features = [x]
        for blocks in self.additional_blocks:
            x = blocks(x)
            features.append(x)

        # Feature Map 38x38x4, 19x19x6, 10x10x6, 5x5x6, 3x3x4, 1x1x4
        locs, confs = self.detection_header(features)

        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs

