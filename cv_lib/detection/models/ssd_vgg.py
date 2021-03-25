from typing import List

from torch import Tensor
from torch.nn import Module

from models.backbones import VGGBackbone
from .sub_module import BoxPredictor


class SSD300_VGG16(Module):
    """
    Only accept 300x300 input image

    Support manual bn layer
    """
    def __init__(
        self,
        backbone: VGGBackbone,
        out_channels: List[int] = [512, 1024, 512, 256, 256, 256],
        num_default_boxes: List[int] = [4, 6, 6, 6, 4, 4],
        n_classes: int = 81
    ):
        super().__init__()

        assert len(out_channels) == len(num_default_boxes) == 6, "only support 6 middle features"
        self.feature_extractor = backbone
        self.detection_header = BoxPredictor(n_classes, out_channels, num_default_boxes)

    def forward(self, x: Tensor):
        features = self.feature_extractor(x)
        # Feature Map 38x38, 19x19, 10x10, 5x5, 3x3, 1x1
        locs, confs = self.detection_header(features)
        # For SSD 300, shall return nbatch x 8732 x {nlabels, nlocs} results
        return locs, confs
