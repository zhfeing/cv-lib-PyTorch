import copy
from typing import Dict, Any

from torch.utils.data import Dataset

from .cifar import CIFAR10, CIFAR100
from .imagenet import ImageNet
from .tiny_imagenet import TinyImagenet


DATASET_DICT = {
    "cifar-10": CIFAR10,
    "cifar-100": CIFAR100,
    "cifar10": CIFAR10,
    "cifar100": CIFAR100,
    "imagenet": ImageNet,
    "tiny-imagenet": TinyImagenet,
}


def get_dataset(dataset_cfg: Dict[str, Any], **kwargs) -> Dataset:
    cfg = copy.deepcopy(dataset_cfg)
    name: str = cfg.pop("name")
    return DATASET_DICT[name.lower()](**cfg, **kwargs)

