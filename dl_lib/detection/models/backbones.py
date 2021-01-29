from torch import Tensor
from torch.nn import Module
from torch.nn.modules.container import Sequential
from torchvision.models.resnet import *
from .vgg_16 import *


__all__ = [
    "Backbone",
    "ResNetBackbone",
    "VGGBackbone"
]


class Backbone(Module):
    """
    Given backbone model, extract specific layer output during forward computation

    Warning:
        Backbone must can be separate into sequence of blocks and extract_layer name must be
        one of its children not sub model
    """
    def __init__(self, backbone: Module, extract_layer_name: str):
        super().__init__()
        self.extract_feature: Tensor = None

        self.partial_backbone = Sequential()
        for name, layer in backbone.named_children():
            self.partial_backbone.add_module(name, layer)
            if name == extract_layer_name:
                break

    def __len__(self) -> int:
        return len(self.partial_backbone)

    def forward(self, x: Tensor) -> Tensor:
        return self.partial_backbone(x)


class ResNetBackbone(Backbone):
    __REGISTERED_BACKBONE__ = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152
    }

    def __init__(self, backbone: str, extract_layer_name: str):
        backbone = self.__REGISTERED_BACKBONE__[backbone](pretrained=True)
        super().__init__(backbone, extract_layer_name)

        conv4_block1 = self.partial_backbone[-1][0]
        conv4_block1.conv1.stride = (1, 1)
        conv4_block1.conv2.stride = (1, 1)
        conv4_block1.downsample[0].stride = (1, 1)


class VGGBackbone(Module):
    __REGISTERED_BACKBONE__ = {
        "vgg16_300": vgg_300,
        "vgg16_512": vgg_512,
    }

    def __init__(self, backbone: str):
        super().__init__()
        self.vgg = self.__REGISTERED_BACKBONE__[backbone](pretrained=True)

    def __len__(self):
        raise Exception("VGG Backbone not Support len")

    def forward(self, x: Tensor) -> Tensor:
        return self.vgg(x)
