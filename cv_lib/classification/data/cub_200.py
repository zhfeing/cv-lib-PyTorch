import os
from typing import Callable, Tuple, Optional, Dict, Any, List

from PIL import Image
import pandas as pd

import torch
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import default_loader

from cv_lib.utils import log_utils

from .classification_dataset import ClassificationDataset
from .imagenet import MEAN, STD


class CUB_200(ClassificationDataset):
    """
    Image folder:
        ├── images
        │   ├── 001.Black_footed_Albatross
        │       ├── Black_Footed_Albatross_0001_796111.jpg
        |   |   ├── ...
        │   ├── 002.Laysan_Albatross
        │   ├── ...
        ├── images.txt
        ├── train_test_split.txt
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
            root: root to Sketches folder
            split: split of dataset, i.e., `train` and `test`
            resize: all images will be resized to given size. If `None`, all images will not be resized
        """
        super().__init__(resize, augmentations)
        self.root = os.path.expanduser(root)
        verify_str_arg(split, "split", ("train", "test"))
        self.split = split
        self.data_folder = os.path.join(self.root, "images")
        self.logger = log_utils.get_master_logger("CUB_200")
        self._init_dataset()

    def _init_dataset(self):
        self.dataset_mean = MEAN
        self.dataset_std = STD

        # handle classes, make label start from 0
        classes = pd.read_csv(
            os.path.join(self.root, "classes.txt"),
            sep=" ",
            names=["target", "name"]
        )
        for label, row in classes.iterrows():
            self.label_info[label] = row["name"]
            self.label_map["name"] = label

        images = pd.read_csv(
            os.path.join(self.root, "images.txt"),
            sep=" ",
            names=["idx", "fp"]
        )
        labels = pd.read_csv(
            os.path.join(self.root, "image_class_labels.txt"),
            sep=" ",
            names=["idx", "target"]
        )
        split = pd.read_csv(
            os.path.join(self.root, "train_test_split.txt"),
            sep=" ",
            names=["idx", "is_training"]
        )
        images = images.merge(split, on="idx").merge(labels, on="idx")
        self.instances = images[split["is_training"] == (self.split == "train")]

    def __len__(self):
        return len(self.instances)

    def get_image(self, index: int) -> Image:
        image_fp = os.path.join(self.data_folder, self.instances["fp"].iloc[index])
        image = default_loader(image_fp)
        return image

    def get_annotation(self, index: int) -> Dict[str, Any]:
        # make label start from 0
        label = self.instances["target"].iloc[index] - 1
        annot = dict(label=torch.tensor(label))
        return annot

