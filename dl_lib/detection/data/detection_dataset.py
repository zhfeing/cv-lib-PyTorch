import os
from typing import Callable, Dict, Tuple, List, Any

from PIL.Image import Image

import torch
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset
from torchvision.datasets.folder import pil_loader
import torchvision.transforms.functional as TF


class DetectionDataset(Dataset):
    """
    Base Detection Dataset

    Required:
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
            resize: Tuple[int] = (300, 300),
            augmentations: Callable[[Image, FloatTensor, LongTensor], Tuple[Image, FloatTensor, LongTensor]] = None,
    ):
        """
        resize: (h, w)
        """
        self.resize = resize
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

    def __getitem__(self, index: int):
        """
        Return image, image_id, img_size (hxw), list of bounding boxes and list of bounding box labels
        Guarantee: bound boxes has more than one elements
        """
        img_id = self.img_keys[index]
        image_name, bboxes = self.images[img_id]

        img = pil_loader(os.path.join(self.img_sub_folder, image_name))
        img_w, img_h = img.size

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

        if self.augmentations is not None:
            img, bbox_sizes, bbox_labels = self.augmentations(img, bbox_sizes, bbox_labels)

        img = TF.resize(img, self.resize)
        img = TF.to_tensor(img)
        img = TF.normalize(img, self.dataset_mean, self.dataset_std, inplace=True)

        info = dict(
            img_id=img_id,
            size=(img_h, img_w)
        )
        info.update(self.other_info(img_id))
        return img, bbox_sizes, bbox_labels, info

