import copy
from typing import Dict, Any, Callable, Tuple

from torch.utils.data import Dataset

from .classification_dataset import ClassificationDataset
from .mnist import MNIST
from .cifar import CIFAR_10, CIFAR_100


__REGISTERED_DATASETS__ = {
    "mnist": MNIST,
    "cifar_10": CIFAR_10,
    "cifar_100": CIFAR_100
}


def get_dataset(
    dataset_cfg: Dict[str, Any],
    train_augmentations: Callable,
    val_augmentations: Callable
) -> Tuple[ClassificationDataset, ClassificationDataset, int]:
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
    return train_dataset, val_dataset, n_classes
