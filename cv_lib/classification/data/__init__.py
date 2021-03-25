from torch.utils.data import Dataset

from .cifar import get_cifar_10, get_cifar_100
from .imagenet import get_imagenet
from .tiny_imagenet import get_tiny_imagenet


DATASET_DICT = {
    "cifar-10": get_cifar_10,
    "cifar-100": get_cifar_100,
    "cifar10": get_cifar_10,
    "cifar100": get_cifar_100,
    "imagenet": get_imagenet,
    "tiny-imagenet": get_tiny_imagenet,
}


def get_dataset(name: str, root: str, split: str = "train") -> Dataset:
    fn = DATASET_DICT[name]
    return fn(root=root, split=split)
