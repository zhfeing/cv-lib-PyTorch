import os
from typing import Callable, Tuple, Optional, Dict, Any, List

from PIL import Image

import torch
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import default_loader

from cv_lib.utils import log_utils

from .classification_dataset import ClassificationDataset
from .utils import make_datafolder


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ImageNet(ClassificationDataset):
    """
    Image folder:
        ├── devkit
        │   └── data
        ├── train_set
        │   ├── n01440764
        │       ├── n01440764_10026.JPEG
        |   |   ├── ...
        │   ├── n01443537
        │   ├── ...
        ├── val_set
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        resize: Optional[Tuple[int]] = None,
        augmentations: Callable[[Image.Image, Dict[str, Any]], Tuple[Image.Image, Dict[str, Any]]] = None,
        make_partial: float = None,
        dataset_mean: List[float] = MEAN,
        dataset_std: List[float] = STD
    ):
        """
        Args:
            root: root to Imagenet folder
            split: split of dataset, i.e., `train` and `val`
            resize: all images will be resized to given size. If `None`, all images will not be resized
        """
        super().__init__(resize, augmentations)
        self.root = os.path.expanduser(root)
        verify_str_arg(split, "split", ("train", "val"))
        self.split = split
        self.data_folder = os.path.join(self.root, self.split)
        self.meta_folder = os.path.join(self.root, "devkit", "data")

        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std

        self.logger = log_utils.get_master_logger("Imagenet")
        self._init_dataset(make_partial)

    def _init_dataset(self, make_partial: float):
        self.logger.info("Reading dataset folder...")
        self.instances, self.label_info, self.label_map = make_datafolder(
            self.data_folder,
            make_partial
        )

    def __len__(self):
        return len(self.instances)

    def get_image(self, index: int) -> Image:
        image_fp = os.path.join(self.data_folder, self.instances[index][0])
        image = default_loader(image_fp)
        return image

    def get_annotation(self, index: int) -> Dict[str, Any]:
        label = self.instances[index][1]
        annot = dict(label=torch.tensor(label))
        return annot
