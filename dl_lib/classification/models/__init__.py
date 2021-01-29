from typing import Dict

import torch

from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet18, ResNet34, ResNet50, ResNet101, ResNet152
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2

MODEL_DICT = {
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet18': ResNet18,
    'ResNet34': ResNet34,
    'ResNet50': ResNet50,
    'ResNet101': ResNet101,
    'ResNet152': ResNet152,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
}


def get_model(model_name: str, num_classes: int, state_dict: Dict[str, torch.Tensor] = None, **kwargs):
    fn = MODEL_DICT[model_name]
    model = fn(num_classes=num_classes, **kwargs)

    if state_dict is not None:
        model.load_state_dict(state_dict)
    return model

