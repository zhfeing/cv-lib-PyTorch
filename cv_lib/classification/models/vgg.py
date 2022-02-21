from typing import Union, List, Dict, Any

import torch
import torch.nn as nn
from torchvision.models.vgg import make_layers


class VGG_Light(nn.Module):

    def __init__(
        self,
        features: nn.Module,
        last_channel: int = 512,
        num_classes: int = 1000,
    ) -> None:
        super().__init__()
        self.features = features
        self.flatten = nn.Flatten(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(last_channel, num_classes)
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    # half
    "HA": [32, "M", 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M"],
    "HB": [32, 32, "M", 64, 64, "M", 128, 128, "M", 256, 256, "M", 256, 256, "M"],
    "HD": [32, 32, "M", 64, 64, "M", 128, 128, 128, "M", 256, 256, 256, "M", 256, 256, 256, "M"],
    "HE": [32, 32, "M", 64, 64, "M", 128, 128, 128, 128, "M", 256, 256, 256, 256, "M", 256, 256, 256, 256, "M"],
}


def _vgg(cfg: str, batch_norm: bool, **kwargs: Any) -> VGG_Light:
    model = VGG_Light(make_layers(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model


def vgg11(**kwargs: Any) -> VGG_Light:
    return _vgg("A", batch_norm=False, **kwargs)


def vgg11_bn(**kwargs: Any) -> VGG_Light:
    return _vgg("A", batch_norm=True, **kwargs)


def vgg13(**kwargs: Any) -> VGG_Light:
    return _vgg("B", batch_norm=False, **kwargs)


def vgg13_bn(**kwargs: Any) -> VGG_Light:
    return _vgg("B", batch_norm=True, **kwargs)


def vgg16(**kwargs: Any) -> VGG_Light:
    return _vgg("D", batch_norm=False, **kwargs)


def vgg16_bn(**kwargs: Any) -> VGG_Light:
    return _vgg("D", batch_norm=True, **kwargs)


def vgg19(**kwargs: Any) -> VGG_Light:
    return _vgg("E", batch_norm=False, **kwargs)


def vgg19_bn(**kwargs: Any) -> VGG_Light:
    return _vgg("E", batch_norm=True, **kwargs)


MODEL_DICT = {
    "light_vgg11": vgg11,
    "light_vgg11_bn": vgg11_bn,
    "light_vgg13": vgg13,
    "light_vgg13_bn": vgg13_bn,
    "light_vgg16": vgg16,
    "light_vgg16_bn": vgg16_bn,
    "light_vgg19": vgg19,
    "light_vgg19_bn": vgg19_bn,
}
