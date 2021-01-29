import json
import os
from typing import Callable, Tuple
import logging

from PIL.Image import Image

from torch import FloatTensor, LongTensor
from torchvision.datasets.utils import verify_str_arg

from .detection_dataset import DetectionDataset


COCO_MEAN = [0.485, 0.456, 0.406]
COCO_STD = [0.229, 0.224, 0.225]


class CocoDetection(DetectionDataset):
    """
    MS Coco Detection <https://cocodataset.org/#detection-2016> Dataset.
    """
    def __init__(
            self,
            root: str,
            split: str = "train",
            version: str = "2017",
            resize: Tuple[int] = (300, 300),
            augmentations: Callable[[Image, FloatTensor, LongTensor], Tuple[Image, FloatTensor, LongTensor]] = None,
    ):
        super().__init__(resize, augmentations)

        self.root = os.path.expanduser(root)
        verify_str_arg(split, "split", ("train", "val"))

        self.logger = logging.getLogger("CocoDetection")

        # path to image folder, e.g. coco_root/train2017
        self.img_sub_folder = os.path.join(self.root, "{}{}".format(split, version))

        # path to annotation, e.g. coco_root/annotations/instances_train2017
        annotation_fp = os.path.join(self.root, "annotations", "instances_{}{}.json".format(split, version))

        self.logger.info("Parsing COCO %s dataset...", split)
        self._init_dataset(annotation_fp)
        self.logger.info("Parsing COCO %s dataset done", split)

    def _init_dataset(self, annotation_fp: str):
        self.logger.info("Reading annotation file...")
        with open(annotation_fp) as f:
            data = json.load(f)
        self.logger.info("Reading annotation file done")
        # 0 stand for the background
        cnt = 0
        self.label_info[cnt] = "background"
        for cat in data["categories"]:
            cnt += 1
            self.label_map[cat["id"]] = cnt
            self.label_info[cnt] = cat["name"]

        # build inference for images
        for img in data["images"]:
            img_id = img["id"]
            img_name = img["file_name"]
            if img_id in self.images:
                raise Exception("duplicated image record")
            self.images[img_id] = (img_name, [])

        # read bboxes
        for bboxes in data["annotations"]:
            img_id = bboxes["image_id"]
            category_id = bboxes["category_id"]
            bbox = bboxes["bbox"]
            bbox_label = self.label_map[category_id]
            self.images[img_id][1].append((bbox, bbox_label))

        for k, v in list(self.images.items()):
            # remove image with no annotations
            if len(v[1]) == 0:
                self.images.pop(k)

        self.img_keys = list(self.images.keys())
        self.dataset_mean = COCO_MEAN
        self.dataset_std = COCO_STD
