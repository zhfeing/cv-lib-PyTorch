from copy import deepcopy
from typing import Dict, Any

import torch
from torch.nn import Module

from .resnet import *
from .cifar_large_resnet import *
from .cifar_small_resnet import *
from .wrn import *
from .vgg import *
from .mobilenetv2 import *


__MODEL_DICT__ = {
    # small resnet for cifar
    "resnet20_cs": resnet20_cs,
    "resnet32_cs": resnet32_cs,
    "resnet56_cs": resnet56_cs,
    # large ResNet for cifar
    "ResNet18_cl": resnet18_cl,
    "ResNet34_cl": resnet34_cl,
    "ResNet50_cl": resnet50_cl,
    # ResNet for large scale dataset
    "ResNet10": resnet10,
    "ResNet18": resnet18,
    "ResNet34": resnet34,
    "ResNet50": resnet50,
    "ResNet101": resnet101,
    "ResNet152": resnet152,
    # wide resnet
    "wrn_16_1": wrn_16_1,
    "wrn_16_2": wrn_16_2,
    "wrn_40_1": wrn_40_1,
    "wrn_40_2": wrn_40_2,
    # vgg
    "vgg11": vgg11_bn,
    "vgg13": vgg13_bn,
    "vgg16": vgg16_bn,
    "vgg19": vgg19_bn,
    # mobile net
    "MobileNetV2": mobile_half,
}


def get_model(model_cfg: Dict[str, Any], num_classes: int):
    model_cfg = deepcopy(model_cfg)
    model_cfg.pop("name")
    name = model_cfg.pop("model_name")
    model: Module = __MODEL_DICT__[name](num_classes=num_classes, **model_cfg)
    return model

