import os
from typing import Callable, Tuple, Optional, Dict, Any

from PIL import Image

import torch
from torchvision.datasets.folder import default_loader

from cv_lib.utils import log_utils

from .classification_dataset import ClassificationDataset
from .imagenet import MEAN, STD
from .utils import make_datafolder


class ProbDatset(ClassificationDataset):
    def __init__(
        self,
        root: str,
        make_partial: float = None,
        resize: Optional[Tuple[int]] = None,
        augmentations: Callable[[Image.Image, Dict[str, Any]], Tuple[Image.Image, Dict[str, Any]]] = None,
        dataset_mean: Tuple[float] = MEAN,
        dataset_std: Tuple[float] = STD
    ):
        """
        Args:
            root: root to probdata folder
            resize: all images will be resized to given size. If `None`, all images will not be resized
        """
        super().__init__(resize, augmentations)
        self.root = os.path.expanduser(root)
        self.data_folder = self.root
        self.logger = log_utils.get_master_logger("Sketches")
        self._init_dataset(dataset_mean, dataset_std, make_partial)

    def _init_dataset(self, dataset_mean: Tuple[float], dataset_std: Tuple[float], make_partial: float):
        self.dataset_mean = dataset_mean
        self.dataset_std = dataset_std
        self.logger.info("Reading dataset folder...")
        self.instances, self.label_info, self.label_map = make_datafolder(self.data_folder, make_partial)

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

