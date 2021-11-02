import os
from typing import Callable, Tuple, Optional, Dict, Any, List

from PIL import Image

import torch
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import pil_loader, is_image_file

from cv_lib.utils import log_utils, random_pick_instances

from .classification_dataset import ClassificationDataset


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class ImageNet(ClassificationDataset):
    """
    Image folder:
        ├── devkit
        │   └── data
        ├── train_set
        │   ├── n01440764
        │       ├── n01440764_10026.JPEG
        |   |   ├── ...
        │   ├── n01443537
        │   ├── ...
        ├── val_set
    """
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

        self.logger = log_utils.get_master_logger("Imagenet")
        self._init_dataset(make_partial)

    def _init_dataset(self, make_partial: float):
        self.dataset_mean = MEAN
        self.dataset_std = STD
        self.logger.info("Reading annotation file...")
        # get classes and class map
        self.CLASSES = [d.name for d in os.scandir(self.data_folder) if d.is_dir()]
        self.CLASSES.sort()
        for i, cls_name in enumerate(self.CLASSES):
            self.label_info[i] = cls_name
            self.label_map[cls_name] = i
        instances_by_class = make_dataset(self.data_folder, self.label_map)
        self.instances: List[Tuple[str, int]] = []
        for i, instances in enumerate(instances_by_class):
            instances = random_pick_instances(instances, make_partial, i)
            self.instances.extend(instances)

    def __len__(self):
        return len(self.instances)

    def get_image(self, index: int) -> Image:
        image_fp = os.path.join(self.data_folder, self.instances[index][0])
        image = pil_loader(image_fp)
        return image

    def get_annotation(self, index: int) -> Dict[str, Any]:
        label = self.instances[index][1]
        annot = dict(label=torch.tensor(label))
        return annot


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
) -> List[List[Tuple[str, int]]]:
    """
    Generates a list of samples of a form (path_to_sample, class).
    Args:
        directory (str): root dataset directory
        class_to_idx (Dict[str, int]): dictionary mapping class name to class index
    Returns:
        List[List[Tuple[str, int]]]: samples of a form (path_to_sample, class) by class_id
    """
    instances_by_class: List[List[Tuple[str, int]]] = []
    directory = os.path.expanduser(directory)

    for target_class in sorted(class_to_idx.keys()):
        instances: List[Tuple[str, int]] = []
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            raise Exception("Class {} has no image files".format(target_class))
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_image_file(path):
                    item = path, class_index
                    instances.append(item)
            instances_by_class.append(instances)
    return instances_by_class

