import collections
from typing import Callable, Tuple, Optional, Dict, Any, List

from PIL import Image
import numpy as np

import torch
import torchvision.datasets
from torchvision.datasets.utils import verify_str_arg

from .classification_dataset import ClassificationDataset
from cv_lib.utils import log_utils, random_pick_instances


CIFAR_10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_10_STD = [0.2023, 0.1994, 0.2010]
CIFAR_100_MEAN = [0.5071, 0.4867, 0.4408]
CIFAR_100_STD = [0.2023, 0.1994, 0.2010]


def _make_partial(data: np.ndarray, targets: List[int], percent: float) -> Tuple[np.ndarray, List[int]]:
    bucket = collections.defaultdict(list)
    for id, c in enumerate(targets):
        bucket[c].append(data[id])

    data: List[np.ndarray] = list()
    targets: List[int] = list()
    for c_id, k in enumerate(bucket):
        instance = random_pick_instances(bucket[k], percent, seed=c_id)
        target = list(k for _ in range(len(instance)))
        data.append(instance)
        targets.extend(target)
    data = np.concatenate(data)
    return data, targets


class CIFAR_10(ClassificationDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        resize: Optional[Tuple[int]] = None,
        augmentations: Callable[[Image.Image, Dict[str, Any]], Tuple[Image.Image, Dict[str, Any]]] = None,
        make_partial: float = None
    ):
        """
        Args:
            root: root to CIFAR folder
            split: split of dataset, i.e., `train` and `test`
            resize: all images will be resized to given size. If `None`, all images will not be resized
        """
        super().__init__(resize, augmentations)
        verify_str_arg(split, "split", ("train", "test"))

        self.logger = log_utils.get_master_logger("CIFAR_10")
        self.split = split
        self.root = root
        self.cifar = torchvision.datasets.CIFAR10(
            root=self.root,
            train=self.split == "train",
            download=False
        )
        self._init_dataset(make_partial)

    def _init_dataset(self, make_partial: float):
        self.CLASSES = self.cifar.classes
        for i, cls_name in enumerate(self.CLASSES):
            self.label_info[i] = cls_name
            self.label_map[cls_name] = i
        self.dataset_mean = CIFAR_10_MEAN
        self.dataset_std = CIFAR_10_STD
        # make partial
        if make_partial is not None:
            data, targets = _make_partial(self.cifar.data, self.cifar.targets, make_partial)
            self.cifar.data = data
            self.cifar.targets = targets

    def __len__(self):
        return len(self.cifar)

    def get_image(self, index: int) -> Image:
        img = self.cifar.data[index]
        return Image.fromarray(img)

    def get_annotation(self, index: int) -> Dict[str, Any]:
        label = int(self.cifar.targets[index])
        annot = dict(label=torch.tensor(label))
        return annot


class CIFAR_100(ClassificationDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        resize: Optional[Tuple[int]] = None,
        augmentations: Callable[[Image.Image, Dict[str, Any]], Tuple[Image.Image, Dict[str, Any]]] = None,
        make_partial: float = None
    ):
        """
        Args:
            root: root to CIFAR folder
            split: split of dataset, i.e., `train` and `test`
            resize: all images will be resized to given size. If `None`, all images will not be resized
        """
        super().__init__(resize, augmentations)
        verify_str_arg(split, "split", ("train", "test"))

        self.logger = log_utils.get_master_logger("CIFAR_100")
        self.split = split
        self.root = root
        self.cifar = torchvision.datasets.CIFAR100(
            root=self.root,
            train=self.split == "train",
            download=False
        )
        self._init_dataset(make_partial)

    def _init_dataset(self, make_partial: float):
        self.CLASSES = self.cifar.classes
        for i, cls_name in enumerate(self.CLASSES):
            self.label_info[i] = cls_name
            self.label_map[cls_name] = i
        self.dataset_mean = CIFAR_100_MEAN
        self.dataset_std = CIFAR_100_STD
        # make partial
        if make_partial is not None:
            data, targets = _make_partial(self.cifar.data, self.cifar.targets, make_partial)
            self.cifar.data = data
            self.cifar.targets = targets

    def __len__(self):
        return len(self.cifar)

    def get_image(self, index: int) -> Image:
        img = self.cifar.data[index]
        return Image.fromarray(img)

    def get_annotation(self, index: int) -> Dict[str, Any]:
        label = int(self.cifar.targets[index])
        annot = dict(label=torch.tensor(label))
        return annot

