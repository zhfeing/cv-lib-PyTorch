from typing import Callable, Tuple, Optional, Dict, Any

from PIL import Image
import numpy as np

import torch
import torchvision.datasets
from torchvision.datasets.utils import verify_str_arg

from .classification_dataset import ClassificationDataset


MNIST_MEAN = [0.0]
MNIST_STD = [1.0]


class MNIST(ClassificationDataset):
    CLASSES = (
        "0 - zero", "1 - one", "2 - two", "3 - three", "4 - four",
        "5 - five", "6 - six", "7 - seven", "8 - eight", "9 - nine"
    )

    def __init__(
        self,
        root: str,
        split: str = "train",
        resize: Optional[Tuple[int]] = None,
        augmentations: Callable[[Image.Image, Dict[str, Any]], Tuple[Image.Image, Dict[str, Any]]] = None,
        expand_channel: bool = False,
        dataset=torchvision.datasets.MNIST
    ):
        """
        Args:
            root: root to MNIST folder
            split: split of dataset, i.e., `train` and `test`
            resize: all images will be resized to given size. If `None`, all images will not be resized
            expand_channel: expand image channels from 1 to 3
        """
        super().__init__(resize, augmentations)
        verify_str_arg(split, "split", ("train", "test"))
        self.img_channels = 1

        self.split = split
        self.root = root
        self.mnist_dataset = dataset(
            root=self.root,
            train=self.split == "train",
            download=True
        )
        self.get_image = self._get_image_c3 if expand_channel else self._get_image_c1
        self._init_dataset()

    def _init_dataset(self):
        for i, cls_name in enumerate(self.CLASSES):
            self.label_info[i] = cls_name
            self.label_map[cls_name] = i
        self.data = self.mnist_dataset.data.numpy()
        self.dataset_mean = MNIST_MEAN
        self.dataset_std = MNIST_STD

    def __len__(self):
        return len(self.mnist_dataset)

    def _get_image_c1(self, index: int) -> Image:
        img = self.data[index]
        return Image.fromarray(img, mode="L")

    def _get_image_c3(self, index: int) -> Image:
        img: np.ndarray = self.data[index]
        img = np.repeat(img[..., None], 3, axis=-1)
        return Image.fromarray(img)

    def get_annotation(self, index: int) -> Dict[str, Any]:
        label = int(self.mnist_dataset.targets[index])
        annot = dict(label=torch.tensor(label))
        return annot


class FashionMNIST(MNIST):
    CLASSES = (
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    )

    def __init__(
        self,
        root: str,
        split: str = "train",
        resize: Optional[Tuple[int]] = None,
        augmentations: Callable[[Image.Image, Dict[str, Any]], Tuple[Image.Image, Dict[str, Any]]] = None,
        expand_channel: bool = False
    ):
        super().__init__(
            root=root,
            split=split,
            resize=resize,
            augmentations=augmentations,
            expand_channel=expand_channel,
            dataset=torchvision.datasets.FashionMNIST
        )
