import copy
from typing import Dict, Any, Callable, Tuple, Optional

from .classification_dataset import ClassificationDataset
from .mnist import MNIST, FashionMNIST
from .cifar import CIFAR_10, CIFAR_100
from .imagenet import ImageNet, ImageNet_A, ImageNet_R
from .sketches import Sketches
from .stanford_cars import StanfordCars
from .cub_200 import CUB_200
from .prob_dataset import ProbDatset
from .caltech_101 import Caltech_101
from .pascal_3d import Pascal3D
from .food_101 import Food_101
from .mini_imagenet import MiniImagenet


__REGISTERED_DATASETS__ = {
    "mnist": MNIST,
    "fashion_mnist": FashionMNIST,
    "cifar_10": CIFAR_10,
    "cifar_100": CIFAR_100,
    "imagenet": ImageNet,
    "imagenet_a": ImageNet_A,
    "imagenet_r": ImageNet_R,
    "sketches": Sketches,
    "stanford_cars": StanfordCars,
    "cub_200": CUB_200,
    "prob_dataset": ProbDatset,
    "caltech_101": Caltech_101,
    "pascal3D": Pascal3D,
    "food_101": Food_101,
    "mini_imagenet": MiniImagenet
}


def get_dataset(
    dataset_cfg: Dict[str, Any],
    train_augmentations: Optional[Callable] = None,
    val_augmentations: Optional[Callable] = None
) -> Tuple[ClassificationDataset, ClassificationDataset, int, int]:
    """
    dataset_cfg:
        {
            dataset: name of dataset
            root: dataset root path
            train:
                xxx: train configs
            val:
                xxx: val configs
            xxx: common configs (for both train and val dictionary)
        }
    """
    # Setup Dataloader
    dataset_cfg = copy.deepcopy(dataset_cfg)
    name = dataset_cfg.pop("name")
    name = name.split("=")[0]
    dataset = __REGISTERED_DATASETS__[name]
    root = dataset_cfg.pop("root")
    train_cfg = dataset_cfg.pop("train")
    val_cfg = dataset_cfg.pop("val")

    train_dataset: ClassificationDataset = dataset(
        root=root,
        augmentations=train_augmentations,
        **train_cfg,
        **dataset_cfg
    )

    val_dataset: ClassificationDataset = dataset(
        root=root,
        augmentations=val_augmentations,
        **val_cfg,
        **dataset_cfg
    )

    assert train_dataset.n_classes == val_dataset.n_classes
    n_classes = train_dataset.n_classes
    img_channels = train_dataset.img_channels
    return train_dataset, val_dataset, n_classes, img_channels


def register_dataset(name: str, dataset: type):
    __REGISTERED_DATASETS__[name] = dataset
