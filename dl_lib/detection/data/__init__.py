from .augmentation import get_composed_augmentations
from .dataloader import get_dataloader
from .detection_dataset import DetectionDataset

from .coco import CocoDetection
from .voc import VOC2012Dataset, VOCPartialDataset


__REGISTERED_DATASETS__ = {
    "COCO": CocoDetection,
    "VOC2007": VOCPartialDataset,
    # "VOC2012": VOC2012Dataset
}


def get_dataset(cfg):
    # Setup Augmentations
    augmentation_list = cfg["training"].get("augmentations", list())
    data_aug = get_composed_augmentations(augmentation_list)

    # Setup Dataloader
    dataset = __REGISTERED_DATASETS__[cfg["data"]["dataset"]]
    data_path = cfg["data"]["path"]

    dataloader_args = cfg["data"].copy()
    dataloader_args.pop('dataset')
    dataloader_args.pop('train_split')
    dataloader_args.pop('val_split')
    dataloader_args.pop('path')

    train_dataset: DetectionDataset = dataset(
        data_path,
        split=cfg["data"]["train_split"],
        augmentations=data_aug,
        **dataloader_args
    )

    val_dataset: DetectionDataset = dataset(
        data_path,
        split=cfg["data"]["val_split"],
        **dataloader_args
    )

    assert(train_dataset.n_classes == val_dataset.n_classes)
    n_classes = train_dataset.n_classes
    return train_dataset, val_dataset, n_classes
