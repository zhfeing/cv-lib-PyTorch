from typing import Callable, Tuple, Optional, Dict, Any

from PIL import Image

import torch
import torchvision.datasets
from torchvision.datasets.utils import verify_str_arg

from .classification_dataset import ClassificationDataset
from cv_lib.utils import log_utils


MNIST_MEAN = [0.5]
MNIST_STD = [1.0]


class MNIST(ClassificationDataset):
    CLASSES = (
        "0 - zero", "1 - one", "2 - two", "3 - three", "4 - four",
        "5 - five", "6 - six", "7 - seven", "8 - eight", "9 - nine"
    )

    def __init__(
        self,
        root: str,
        split: str = "trainval",
        resize: Optional[Tuple[int]] = None,
        augmentations: Callable[[Image.Image, Dict[str, Any]], Tuple[Image.Image, Dict[str, Any]]] = None,
    ):
        """
        Args:
            root: root to MNIST folder
            split: split of dataset, i.e., `train` and `test`
            resize: all images will be resized to given size. If `None`, all images will not be resized
        """
        super().__init__(resize, augmentations)
        verify_str_arg(split, "split", ("train", "test"))

        self.logger = log_utils.get_master_logger("MNIST")
        self.split = split
        self.root = root
        self.mnist_dataset = torchvision.datasets.MNIST(
            root=self.root,
            train=self.split == "train",
            download=True
        )

    def _init_dataset(self):
        for i, cls_name in enumerate(self.CLASSES):
            self.label_info[i] = cls_name
            self.label_map[cls_name] = i

        self.dataset_mean = MNIST_MEAN
        self.dataset_std = MNIST_STD

    def __len__(self):
        return len(self.mnist_dataset)

    def get_image(self, index: int) -> Image:
        img = self.mnist_dataset.data[index]
        return Image.fromarray(img, mode="L")

    def get_annotation(self, index: int) -> Dict[str, Any]:
        label = int(self.mnist_dataset.targets[index])
        annot = dict(label=torch.tensor(label))
        return annot
