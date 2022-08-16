import os
import json
from typing import Callable, Tuple, Optional, Dict, Any, List

from PIL import Image

import torch
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import default_loader

from .classification_dataset import ClassificationDataset
from .imagenet import MEAN, STD


class Food_101(ClassificationDataset):
    """
    Image folder:
        ├── images
        │   ├── apple_pie
        │       ├── 1005649.jpg
        |   |   ├── ...
        │   ├── baby_back_ribs
        │   ├── ...
        ├── meta
        │   ├── classes.txt
        │   ├── test.json
        │   ├── train.json
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
            root: root to Imagenet folder
            split: split of dataset, i.e., `train` and `val`
            resize: all images will be resized to given size. If `None`, all images will not be resized
        """
        super().__init__(resize, augmentations)
        self.root = os.path.expanduser(root)
        verify_str_arg(split, "split", ("train", "test"))
        self.split = split
        self.data_folder = os.path.join(self.root, "images")
        self._init_dataset()

    def _init_dataset(self):
        self.dataset_mean = MEAN
        self.dataset_std = STD

        meta_path = os.path.join(self.root, "meta")
        cls_fp = os.path.join(meta_path, "classes.txt")
        with open(cls_fp, "r") as f:
            classes = f.readlines()
        classes = [c.strip("\r\n") for c in classes]
        for c_id, c_name in enumerate(classes):
            self.label_info[c_id] = c_name
            self.label_map[c_name] = c_id

        images_fp = os.path.join(meta_path, f"{self.split}.json")
        with open(images_fp, "r") as f:
            image_meta = json.load(f)

        image_folder = os.path.join(self.root, "images")
        instances: List[Tuple[str, int]] = list()
        for c_id, c_name in enumerate(classes):
            img_list: List[str] = image_meta[c_name]
            img_list = [(os.path.join(image_folder, f"{p}.jpg"), c_id) for p in image_meta[c_name]]
            instances.extend(img_list)
        self.instances = instances

    def __len__(self):
        return len(self.instances)

    def get_image(self, index: int) -> Image:
        image_fp = self.instances[index][0]
        image = default_loader(image_fp)
        return image

    def get_annotation(self, index: int) -> Dict[str, Any]:
        label = self.instances[index][1]
        annot = dict(label=torch.tensor(label))
        return annot

