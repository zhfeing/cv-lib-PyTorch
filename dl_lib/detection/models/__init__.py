from functools import partial
from typing import Dict
import copy

from torch.nn import Module
from torchvision.models.resnet import *

from .ssd_resnet import SSD300_ResNet
from .ssd_vgg import SSD300_VGG16
from .backbones import *


__REGISTERED_MODELS__ = {
    "SSD300_ResNet": SSD300_ResNet,
    "SSD300_VGG16": SSD300_VGG16
}

__REGISTERED_BACKBONES__ = {
    "ResNetBackbone": ResNetBackbone,
    "VGGBackbone": VGGBackbone
}


def _get_model_instance(name):
    try:
        return __REGISTERED_MODELS__[name]
    except:
        raise Exception("Model {} not available".format(name))


def get_model_partial(model_cfg, n_classes: int) -> partial:
    model_dict: Dict = copy.deepcopy(model_cfg)
    name = model_dict.pop("arch")
    model = _get_model_instance(name)
    return partial(model, n_classes=n_classes, **model_dict)


def get_backbone(backbone_config) -> Backbone:
    backbone_dict = copy.deepcopy(backbone_config)
    t = backbone_dict.pop("type")
    return __REGISTERED_BACKBONES__[t](**backbone_dict)

