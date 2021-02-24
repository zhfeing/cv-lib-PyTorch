from typing import Callable, Dict, Any, Tuple
from .dataloader import get_dataloader
from .detection_dataset import DetectionDataset

from .coco import CocoDetection
from .voc import VOC0712Dataset, VOC2007Dataset


__REGISTERED_DATASETS__ = {
    "COCO": CocoDetection,
    "VOC2007": VOC2007Dataset,
    "VOC2012": VOC0712Dataset
}


def get_dataset(
    dataset_cfg: Dict[str, Any],
    train_augmentations: Callable,
    val_augmentations: Callable
) -> Tuple[DetectionDataset, DetectionDataset, int]:
    """
    dataset_cfg:
        {
            dataset: name of dataset
            root: dataset root path
            train:
                xxx: train configs
            val:
                xxx: val configs
        }
    """
    # Setup Dataloader
    dataset = __REGISTERED_DATASETS__[dataset_cfg["name"]]
    root = dataset_cfg["root"]

    train_cfg = dataset_cfg["train"]
    val_cfg = dataset_cfg["val"]

    train_dataset: DetectionDataset = dataset(
        root,
        augmentations=train_augmentations,
        **train_cfg
    )

    val_dataset: DetectionDataset = dataset(
        root,
        augmentations=val_augmentations,
        **val_cfg
    )

    assert train_dataset.n_classes == val_dataset.n_classes
    n_classes = train_dataset.n_classes
    return train_dataset, val_dataset, n_classes
