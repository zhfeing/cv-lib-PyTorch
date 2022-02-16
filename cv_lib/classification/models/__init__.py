from copy import deepcopy
from typing import Callable, Dict, Any

from torch.nn import Module


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

