import os
from typing import Callable, Tuple, Optional, Dict, Any

from PIL import Image

import torch
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import default_loader

from cv_lib.utils import log_utils

from .classification_dataset import ClassificationDataset
from .imagenet import MEAN, STD
from .utils import make_datafolder


class Caltech_256(ClassificationDataset):
    """
    Image folder:
        ├── train
        │   ├── 001.ak47
        │       ├── 001_0002.jpg
        |   |   ├── ...
        │   ├── 002.american-flag
        │   ├── ...
        ├── val
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        resize: Optional[Tuple[int]] = None,
        augmentations: Callable[[Image.Image, Dict[str, Any]], Tuple[Image.Image, Dict[str, Any]]] = None
    ):
        """
        Args:
            root: root to Caltech-256 folder
            split: split of dataset, i.e., `train` and `val`
            resize: all images will be resized to given size. If `None`, all images will not be resized
        """
        super().__init__(resize, augmentations)
        self.root = os.path.expanduser(root)
        verify_str_arg(split, "split", ("train", "val"))
        self.split = split
        self.data_folder = os.path.join(self.root, self.split)
        self.logger = log_utils.get_master_logger("Caltech_256")
        self._init_dataset()

    def _init_dataset(self):
        self.dataset_mean = MEAN
        self.dataset_std = STD
        self.logger.info("Reading dataset folder...")
        self.instances, self.label_info, self.label_map = make_datafolder(self.data_folder)

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

