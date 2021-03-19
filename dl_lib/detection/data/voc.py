import enum
import os
from typing import Callable, Tuple, List, Optional, Dict, Any
import logging
import xml.etree.ElementTree as ET
import collections

import tqdm
from PIL.Image import Image

import torch
from torchvision.datasets.utils import verify_str_arg

from .detection_dataset import DetectionDataset


VOC_MEAN = [0.485, 0.456, 0.406]
VOC_STD = [0.229, 0.224, 0.225]


class VOCPartialDataset(DetectionDataset):
    """
    Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/> Detection Dataset.
    Only support 2007 and 2012 datasets
    """
    CLASSES = (
        "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair",
        "cow", "diningtable", "dog", "horse",
        "motorbike", "person", "pottedplant",
        "sheep", "sofa", "train", "tvmonitor"
    )

    def __init__(
            self,
            root: str,
            split: str = "trainval",
            version: str = "2007",
            resize: Optional[Tuple[int]] = (300, 300),
            augmentations: Callable[[Image, Dict[str, Any]], Tuple[Image, Dict[str, Any]]] = None,
            keep_difficult: bool = False
    ):
        super().__init__(resize, augmentations)
        self.keep_difficult = keep_difficult

        verify_str_arg(version, "version", ("2007", "2012"))
        verify_str_arg(split, "split", ("train", "val", "test", "trainval"))

        self.logger = logging.getLogger("VOCDetection")
        self.version = version
        # parse folders
        self.root = os.path.expanduser(os.path.join(root, f"VOC{version}"))
        # read split file
        split_fp = os.path.join(self.root, "ImageSets", "Main", f"{split}.txt")
        if not os.path.isfile(split_fp):
            raise FileNotFoundError(f"`{split_fp}` is not found, note that there is no `test.txt` for VOC-2012")
        with open(split_fp, "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]

        self.logger.info("Parsing VOC%s %s dataset...", version, split)
        self._init_dataset()
        self.logger.info("Parsing VOC%s %s dataset done", version, split)

    def _init_dataset(self):
        # path to image folder, e.g. VOC2007/train2017
        image_folder = os.path.join(self.root, "JPEGImages")
        annotation_folder = os.path.join(self.root, "Annotations")

        # skip 0 for background
        for cls_id, cat in enumerate(self.CLASSES):
            cls_id = cls_id + 1
            self.label_map[cat] = cls_id
            self.label_info[cls_id] = cat

        # build inference for images
        self.images = [os.path.join(image_folder, x + ".jpg") for x in self.file_names]
        self.targets = [os.path.join(annotation_folder, x + ".xml") for x in self.file_names]

        self.dataset_mean = VOC_MEAN
        self.dataset_std = VOC_STD

    def parse_voc_xml(self, annotation_fp: str):
        objects = ET.parse(annotation_fp).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            bbox = obj.find('bndbox')
            # VOC dataset format follows Matlab, in which indexes start from 0
            x1 = int(bbox.find('xmin').text) - 1
            y1 = int(bbox.find('ymin').text) - 1
            x2 = int(bbox.find('xmax').text) - 1
            y2 = int(bbox.find('ymax').text) - 1
            # convert to xywh
            boxes.append([x1, y1, x2 - x1, y2 - y1])
            labels.append(self.label_map[class_name])
            is_difficult.append(int(obj.find('difficult').text))

        boxes = torch.tensor(boxes, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        is_difficult = torch.tensor(is_difficult, dtype=torch.bool)
        return boxes, labels, is_difficult

    def get_img_id(self, index: int) -> str:
        return self.file_names[index]

    def get_annotation(self, index: int) -> Dict[str, Any]:
        # read bboxes
        boxes, labels, is_difficult = self.parse_voc_xml(self.targets[index])
        if not self.keep_difficult:
            boxes = boxes[~is_difficult]
            labels = labels[~is_difficult]
            is_difficult = is_difficult[~is_difficult]
        target = {
            "boxes": boxes,
            "labels": labels,
            "is_difficult": is_difficult
        }
        return target


class VOC2007Dataset(VOCPartialDataset):
    def __init__(
            self,
            root: str,
            split: str = "trainval",
            resize: Optional[Tuple[int]] = (300, 300),
            augmentations: Callable[[Image, Dict[str, Any]], Tuple[Image, Dict[str, Any]]] = None,
    ):
        super().__init__(root, split, version="2007", resize=resize, augmentations=augmentations)


class VOC0712Dataset(VOCPartialDataset):
    """
    Combined VOC2007 and VOC2012 partial
    """
    def __init__(
            self,
            root: str,
            split: str = "trainval",
            split_07: Optional[str] = "trainval",
            split_12: Optional[str] = "trainval",
            resize: Optional[Tuple[int]] = (300, 300),
            augmentations: Callable[[Image, Dict[str, Any]], Tuple[Image, Dict[str, Any]]] = None,
    ):
        sub_datasets: List[VOCPartialDataset] = list()
        if split == "test":
            sub_datasets += [VOC2007Dataset(root, split, resize, augmentations)]
        else:
            sub_datasets += [
                VOC2007Dataset(root, split_07, resize, augmentations),
                VOCPartialDataset(root, split_12, "2012", resize, augmentations)
            ]

        # adaption with `VOCPartialDataset`
        self.resize = tuple(resize) if isinstance(resize, list) else resize
        self.augmentations = augmentations

        self.root = os.path.expanduser(root)

        self.dataset_mean = sub_datasets[0].dataset_mean
        self.dataset_std = sub_datasets[0].dataset_std
        self.label_map = sub_datasets[0].label_map
        self.label_info = sub_datasets[0].label_info
        self.keep_difficult = sub_datasets[0].keep_difficult

        self.images: List[str] = list()
        self.targets: List[str] = list()
        self.file_names: List[str] = list()
        for d in sub_datasets:
            self.images.extend(d.images)
            self.targets.extend(d.targets)
            self.file_names.extend(d.file_names)
