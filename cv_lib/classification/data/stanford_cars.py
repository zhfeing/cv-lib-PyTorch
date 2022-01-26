import os
from typing import Callable, Tuple, Optional, Dict, Any

from PIL import Image
import scipy.io as sio

import torch
from torchvision.datasets.folder import default_loader

from .classification_dataset import ClassificationDataset
from .imagenet import MEAN, STD


class StanfordCars(ClassificationDataset):
    """
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
        self.dataset_mean = MEAN
        self.dataset_std = STD

        training = split == "train"
        loaded_mat = sio.loadmat(os.path.join(self.root, "cars_annos.mat"))

        # define classes
        class_data = loaded_mat["class_names"][0]
        for i, c_name in enumerate(class_data):
            c_name = c_name[0]
            self.label_map[c_name] = i
            self.label_info[i] = c_name

        # collect images
        data = loaded_mat["annotations"][0]
        self.instances: list[Tuple[str, int]] = list()
        for item in data:
            if training != bool(item[-1][0]):
                path = str(item[0][0])
                label = int(item[-2][0]) - 1
                self.instances.append((path, label))

    def __len__(self):
        return len(self.instances)

    def get_image(self, index: int) -> Image:
        image_fp = os.path.join(self.root, self.instances[index][0])
        image = default_loader(image_fp)
        return image

    def get_annotation(self, index: int) -> Dict[str, Any]:
        label = self.instances[index][1]
        annot = dict(label=torch.tensor(label))
        return annot
