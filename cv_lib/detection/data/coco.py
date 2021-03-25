from collections import defaultdict
import json
import os
from typing import Callable, Tuple

import tqdm
from PIL.Image import Image

import torch
from torch import FloatTensor, LongTensor
from torchvision.datasets.utils import verify_str_arg

from .detection_dataset import DetectionDataset
from dl_lib.utils import log_utils


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
            keep_crowd: bool = False
    ):
        super().__init__(resize, augmentations)

        self.root = os.path.expanduser(root)
        verify_str_arg(split, "split", ("train", "val"))
        self.keep_crowd = keep_crowd

        self.logger = log_utils.get_master_logger("CocoDetection")

        # path to image folder, e.g. coco_root/train2017
        self.image_folder = os.path.join(self.root, f"{split}{version}")

        # path to annotation, e.g. coco_root/annotations/instances_train2017
        annotation_fp = os.path.join(self.root, "annotations", f"instances_{split}{version}.json")

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
            self.images[img_id] = os.path.join(self.image_folder, img_name)
            self.annotations[img_id] = defaultdict(list)

        # read annotations
        self.logger.info("Loading annotations...")
        for bbox in tqdm.tqdm(data["annotations"]):
            img_id = bbox["image_id"]
            category_id = bbox["category_id"]
            bbox_label = self.label_map[category_id]
            # 3 keys
            self.annotations[img_id]["boxes"].append(bbox["bbox"])
            self.annotations[img_id]["labels"].append(bbox_label)
            self.annotations[img_id]["iscrowd"].append(bbox["iscrowd"])

        # transformation
        for img_id in tqdm.tqdm(list(self.images.keys())):
            # transform to tensor
            boxes = torch.tensor(self.annotations[img_id]["boxes"], dtype=torch.float).reshape(-1, 4)
            labels = torch.tensor(self.annotations[img_id]["labels"], dtype=torch.long)
            iscrowd = torch.tensor(self.annotations[img_id]["iscrowd"], dtype=torch.bool)
            if not self.keep_crowd:
                boxes = boxes[~iscrowd]
                labels = labels[~iscrowd]
                iscrowd = iscrowd[~iscrowd]
            keep = torch.logical_and(boxes[:, 3] > 0, boxes[:, 2] > 0)
            self.annotations[img_id]["boxes"] = boxes[keep]
            self.annotations[img_id]["labels"] = labels[keep]
            self.annotations[img_id]["iscrowd"] = iscrowd[keep]
            # remove image with no annotations
            if len(self.annotations[img_id]["boxes"]) == 0:
                self.images.pop(img_id)
                self.annotations.pop(img_id)

        self.img_ids = sorted(list(self.images.keys()))
        self.dataset_mean = COCO_MEAN
        self.dataset_std = COCO_STD
