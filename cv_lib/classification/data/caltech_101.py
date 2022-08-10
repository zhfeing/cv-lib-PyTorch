import os
from typing import Callable, Tuple, Optional, Dict, Any

from PIL import Image
import pandas as pd

import torch
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import default_loader

from cv_lib.utils import log_utils

from .classification_dataset import ClassificationDataset
from .imagenet import MEAN, STD
from .utils import make_datafolder


class Caltech_101(ClassificationDataset):
    """
    Image folder:
        ├── train
        │   ├── cat
        │       ├── 10.png
        |   |   ├── ...
        │   ├── 'alarm clock'
        │   ├── ...
        ├── test
    """
    def __init__(
        self,
        root: str,
        split: str = "train",
        resize: Optional[Tuple[int]] = None,
        augmentations: Callable[[Image.Image, Dict[str, Any]], Tuple[Image.Image, Dict[str, Any]]] = None,
        make_partial: float = None,
        manual_classes_fp: str = None,
    ):
        """
        Args:
            root: root to Caltech_101 folder
            split: split of dataset, i.e., `train` and `test`
            resize: all images will be resized to given size. If `None`, all images will not be resized
        """
        super().__init__(resize, augmentations)
        self.root = os.path.expanduser(root)
        verify_str_arg(split, "split", ("train", "test"))
        self.split = split
        self.data_folder = os.path.join(self.root, self.split)
        self.logger = log_utils.get_master_logger("Sketches")

        classes = None
        if manual_classes_fp:
            df = pd.read_csv(manual_classes_fp)
            classes = list(df["classes"])
        self._init_dataset(make_partial, classes)

    def _init_dataset(self, make_partial, manual_classes):
        self.dataset_mean = MEAN
        self.dataset_std = STD
        self.logger.info("Reading dataset folder...")
        self.instances, self.label_info, self.label_map = make_datafolder(
            self.data_folder,
            make_partial,
            manual_classes
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

