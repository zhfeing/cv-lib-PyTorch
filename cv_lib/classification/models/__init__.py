from copy import deepcopy
from typing import Callable, Dict, Any

from torch.nn import Module

from .cifar_large_resnet import MODEL_DICT as cl_models
from .cifar_small_resnet import MODEL_DICT as cs_models
from .resnet import MODEL_DICT as resnets
from .wrn import MODEL_DICT as wrns
from .vgg import MODEL_DICT as vggs
from .alexnet import MODEL_DICT as alexnets


__MODEL_DICT__ = {}


def register_model(name: str, model_fn: Callable[[], Module]):
    assert name not in __MODEL_DICT__, f"model {name} already exists"
    __MODEL_DICT__[name] = model_fn


def register_models(models: Dict[str, Callable[[], Module]]):
    intersection = __MODEL_DICT__.keys() & models
    assert len(intersection) == 0, f"model {intersection} already exists"
    __MODEL_DICT__.update(models)


def get_model(model_cfg: Dict[str, Any], num_classes: int):
    model_cfg = deepcopy(model_cfg)
    model_cfg.pop("name")
    name = model_cfg.pop("model_name")
    model: Module = __MODEL_DICT__[name](num_classes=num_classes, **model_cfg)
    return model


register_models(cl_models)
register_models(cs_models)
register_models(resnets)
register_models(wrns)
register_models(vggs)
register_models(alexnets)
