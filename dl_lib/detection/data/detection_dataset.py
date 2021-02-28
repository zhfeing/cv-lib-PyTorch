import os
from typing import Callable, Dict, Tuple, List, Any, Optional

from PIL.Image import Image

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
import torchvision.transforms.functional as TF
from torchvision.ops.boxes import box_area


class DetectionDataset(Dataset):
    """
    Base Detection Dataset

    Inhered Class Requirements:
        `img_sub_folder`: root folder to all images
        `images`: dict e.g. {img_id: [image_name, bboxes]},
            where bboxes: ((left, top, width, height), bbox_label), you should guardentee that
            at least one bbox for a image and "img_sub_folder/image_name" is the image filepath
        `label_map`: Dict[str, int], map a str label to its id != 0 (for train)
        `label_info`: Dict[int, str], map a label id to its str, 0 for background
        `img_keys`: list of img_id, the `img_id` is the unique id for each image (could be str or int)
        `dataset_mean`: List[float]
        `dataset_std`: List[float]
    """
    def __init__(
            self,
            resize: Optional[Tuple[int]] = (300, 300),
            augmentations: Callable[[Image, Dict[str, Any]], Tuple[Image, Dict[str, Any]]] = None,
    ):
        """
        resize: (h, w)
        """
        self.resize = tuple(resize) if isinstance(resize, list) else resize
        self.augmentations = augmentations

        self.img_sub_folder: str = None
        self.images = dict()
        self.label_map: Dict[str, int] = dict()
        self.label_info: Dict[int, str] = dict()
        self.img_keys = list()
        self.dataset_mean: List[float] = None
        self.dataset_std: List[float] = None

    @property
    def n_classes(self) -> int:
        return len(self.label_info)

    def __len__(self) -> int:
        return len(self.img_keys)

    def other_info(self, img_id: int) -> Dict[str, Any]:
        return dict()

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Any]]:
        """
        Return image and target where target is a dictionary e.g.
            target: {
                image_id: str or int,
                orig_size: original image size (h, w)
                size: image size after transformation (h, w)
                boxes: relative bounding box for each object in the image (x1, y1, x2, y2) [0, 1]
                area: bounding box relative area [0, 1]
                labels: label for each bounding box
                (expand to dict) OTHER_INFO: other information from inhered class `other_info(img_id)`
            }
        Guarantee: bound boxes has more than one elements
        """
        img_id = self.img_keys[index]
        image_name, bboxes = self.images[img_id]

        img = pil_loader(os.path.join(self.img_sub_folder, image_name))
        img_w, img_h = img.size

        target: Dict[str, Any] = {
            "image_id": img_id,
            "orig_size": (img_h, img_w),
            "size": (img_h, img_w)
        }
        target.update(self.other_info(img_id))

        bbox_sizes = []
        bbox_labels = []

        for (x, y, w, h), bbox_label in bboxes:
            right = x + w
            bottom = y + h
            # normalize
            bbox_size = (x / img_w, y / img_h, right / img_w, bottom / img_h)
            bbox_sizes.append(bbox_size)
            bbox_labels.append(bbox_label)

        bbox_sizes = torch.tensor(bbox_sizes, dtype=torch.float)
        bbox_labels = torch.tensor(bbox_labels, dtype=torch.long)

        target["boxes"] = bbox_sizes
        target["labels"] = bbox_labels

        if self.augmentations is not None:
            img, target = self.augmentations(img, target)

        if self.resize is not None:
            img = TF.resize(img, self.resize)
            target["size"] = self.resize
        target["area"] = box_area(target["boxes"])
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.dataset_mean, self.dataset_std, inplace=True)
        return img, target

