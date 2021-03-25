from typing import List

import torch
from torch import Tensor
import torch.nn as nn

from models.aux_layers import conv3x3


__all__ = ["BoxPredictor"]


class BoxPredictor(nn.Module):
    def __init__(self, n_classes: int, out_channels: List[int], num_default_boxes: List[int]):
        super().__init__()
        self.n_classes = n_classes
        self.conf = nn.ModuleList()
        self.loc = nn.ModuleList()

        # n_default_box per pixel of each extracted featuremap
        for n_default_box, out_c in zip(num_default_boxes, out_channels):
            # 4 parameters for bbox offset
            self.loc.append(conv3x3(out_c, n_default_box * 4, bias=True))
            # classification
            self.conf.append(conv3x3(out_c, n_default_box * n_classes, bias=True))

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features: List[Tensor]):
        ret = []
        for x, loc, conf in zip(features, self.loc, self.conf):
            batch_size = x.shape[0]
            locs = loc(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 4)
            confs = conf(x).permute(0, 2, 3, 1).contiguous().view(batch_size, -1, self.n_classes)
            ret.append((locs, confs))

        locs, confs = zip(*ret)
        locs = torch.cat(locs, 1).permute(0, 2, 1).contiguous()
        confs = torch.cat(confs, 1).permute(0, 2, 1).contiguous()
        return locs, confs
