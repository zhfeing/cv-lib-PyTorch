import os
from typing import Callable, Tuple, List, Optional, Dict, Any
import xml.etree.ElementTree as ET

from PIL.Image import Image

import torch
from torchvision.datasets.utils import verify_str_arg

from .detection_dataset import DetectionDataset
from cv_lib.utils import log_utils


VOC_MEAN = [0.485, 0.456, 0.406]
VOC_STD = [0.229, 0.224, 0.225]


class VOCBaseDataset(DetectionDataset):
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
        keep_difficult: bool = False,
        make_partial: List[int] = None
    ):
        """
        Args:
            root: root to voc path which contains [`VOC2007`  `VOC2012`] folders
            split: split of dataset, e.g. `train`, `val`, `test` and `trainval`. Warning: VOC2012 has no
                test split
            version: `2007` or `2012`
            resize: all images will be resized to given size. If `None`, all images will not be resized
            make_partial: only keep objects with given classes, w.r.t `self.CLASSES`. If `None`, all
                objects will be preserved. For multitask learning which one task has 10 classes and the
                other task has others classes
        """
        super().__init__(resize, augmentations)
        self.keep_difficult = keep_difficult

        verify_str_arg(version, "version", ("2007", "2012"))
        verify_str_arg(split, "split", ("train", "val", "test", "trainval", "check"))

        self.logger = log_utils.get_master_logger("VOCDetection")
        self.version = version
        self.split = split
        # parse folders
        self.root = os.path.expanduser(os.path.join(root, f"VOC{version}"))
        # read split file
        split_fp = os.path.join(self.root, "ImageSets", "Main", f"{split}.txt")
        if not os.path.isfile(split_fp):
            raise FileNotFoundError(f"`{split_fp}` is not found, note that there is no `test.txt` for VOC-2012")
        with open(split_fp, "r") as f:
            self.file_names = [x.strip() for x in f.readlines()]

        self.logger.info("Parsing VOC%s %s dataset...", version, split)
        self._init_dataset(make_partial)
        self.logger.info("Parsing VOC%s %s dataset done", version, split)

    def _init_dataset(self, make_partial: List[int] = None):
        # path to image folder, e.g. VOC2007/train2017
        image_folder = os.path.join(self.root, "JPEGImages")
        annotation_folder = os.path.join(self.root, "Annotations")

        if make_partial is not None:
            make_partial.sort()
            self.CLASSES = tuple(self.CLASSES[c] for c in make_partial)
            self.logger.info("Partial VOC%s %s dataset with classes: %s", self.version, self.split, str(self.CLASSES))

        # skip 0 for background
        for cls_id, cat in enumerate(self.CLASSES):
            cls_id = cls_id + 1
            self.label_map[cat] = cls_id
            self.label_info[cls_id] = cat

        # build inference for images
        self.images = [os.path.join(image_folder, f"{x}.jpg") for x in self.file_names]
        self.targets = [os.path.join(annotation_folder, f"{x}.xml") for x in self.file_names]

        self.dataset_mean = VOC_MEAN
        self.dataset_std = VOC_STD

    def parse_voc_xml(self, annotation_fp: str):
        objects = ET.parse(annotation_fp).findall("object")
        boxes = []
        labels = []
        is_difficult = []
        for obj in objects:
            class_name = obj.find('name').text.lower().strip()
            # remove abandoned classes
            if class_name not in self.CLASSES:
                continue
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

        boxes = torch.tensor(boxes, dtype=torch.float).reshape(-1, 4)
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

    def set_samples(self, keep_ids: List[str]):
        samples: List[int] = list()
        keep_ids = set(keep_ids)
        for idx, img_id in enumerate(self.file_names):
            if img_id in keep_ids:
                samples.append(idx)
        samples.sort()
        self.samples = samples


class VOC2007Dataset(VOCBaseDataset):
    def __init__(self, **kwargs):
        super().__init__(version="2007", **kwargs)


class VOC0712Dataset(VOCBaseDataset):
    """
    Combined VOC2007 and VOC2012 partial
    """
    def __init__(
            self,
            split: str = "trainval",
            split_07: Optional[str] = "trainval",
            split_12: Optional[str] = "trainval",
            **kwargs
    ):
        sub_datasets: List[VOCBaseDataset] = list()
        if split == "test":
            sub_datasets += [VOC2007Dataset(split=split, **kwargs)]
        else:
            sub_datasets += [
                VOC2007Dataset(split=split_07, **kwargs),
                VOCBaseDataset(split=split_12, version="2012", **kwargs)
            ]

        # adaption with `VOCPartialDataset`
        resize = kwargs["resize"]
        augmentations = kwargs["augmentations"]
        root = kwargs["root"]

        self.resize = tuple(resize) if isinstance(resize, list) else resize
        self.augmentations = augmentations

        self.root = os.path.expanduser(root)

        self.CLASSES = sub_datasets[0].CLASSES
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
            assert self.CLASSES == d.CLASSES, "all sub dataset must have the same `CLASSES`"
