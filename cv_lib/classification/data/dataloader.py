from typing import Tuple, List, Dict, Any

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


InfoType = Dict[str, Any]


def detection_collate(batch) -> Tuple[Tensor, List[Tensor], List[Tensor], List[InfoType]]:
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (Tensor) batch of images stacked on their 0 dim
            2) (List[Tensor]) batch of bbox_sizes
            3) (List[Tensor]) batch of bbox_labels
            4) (List[InfoType]) batch of image infomation
    """
    imgs, bbox_sizes, bbox_labels, img_info = zip(*batch)
    return torch.stack(imgs), list(bbox_sizes), list(bbox_labels), list(img_info)


def get_dataloader(cfg, train_dataset: Dataset, val_dataset: Dataset) -> DataLoader:
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["training"]["n_workers"],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        collate_fn=detection_collate
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["validation"]["batch_size"],
        num_workers=cfg["validation"]["n_workers"],
        shuffle=False,
        pin_memory=True,
        collate_fn=detection_collate
    )
    return train_loader, val_loader
