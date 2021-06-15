import collections
import json
import os
from typing import Callable, DefaultDict, Tuple, List, Dict, Any

import tqdm
from PIL.Image import Image
import numpy as np

import torch
from torch import FloatTensor, LongTensor
from torchvision.datasets.utils import verify_str_arg

from .detection_dataset import DetectionDataset
from cv_lib.utils import log_utils, load_object, save_object
import cv_lib.distributed.utils as dist_utils


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
            keep_crowd: bool = False,
            make_partial: List[int] = None,
            fast_record_fp: str = None,
            keep_percent: float = None
    ):
        super().__init__(resize, augmentations)

        self.root = os.path.expanduser(root)
        verify_str_arg(version, "version", ("2017"))
        verify_str_arg(split, "split", ("train", "val"))

        self.logger = log_utils.get_master_logger("CocoDetection")
        self.version = version
        self.split = split
        self.keep_crowd = keep_crowd
        self.make_partial = make_partial

        # path to image folder, e.g. coco_root/train2017
        self.image_folder = os.path.join(self.root, f"{split}{version}")

        # path to annotation, e.g. coco_root/annotations/instances_train2017
        annotation_fp = os.path.join(self.root, "annotations", f"instances_{split}{version}.json")

        self.logger.info("Parsing COCO %s dataset...", split)
        self._init_dataset(annotation_fp, make_partial, fast_record_fp)
        self.logger.info("Parsing COCO %s dataset done", split)

        if keep_percent is not None:
            self.logger.info("Reduce COCO {} dataset to %{:.2f}".format(split, keep_percent))
            self._keep_percent(keep_percent)

    def _keep_percent(self, keep_percent: float):
        n_total = len(self.image_ids)
        n_keep = int(n_total * keep_percent / 100)
        if n_keep == 0:
            raise "keep_percent ({}) is too small".format(keep_percent)
        self.logger.info("COCO %s dataset keep %d instances", self.split, n_keep)
        rng = np.random.default_rng(12345)
        perm = rng.permutation(n_total)[:n_keep]
        perm.sort()
        self.image_ids = np.asarray(self.image_ids)[perm].tolist()
        self.images = np.asarray(self.images)[perm].tolist()

    def _check_record(self, record: Dict[str, Any]) -> bool:
        p1 = record["make_partial"] == self.make_partial
        p2 = record["version"] == self.version
        p3 = record["split"] == self.split
        p4 = record["keep_crowd"] == self.keep_crowd
        return p1 and p2 and p3 and p4

    def _init_dataset(
        self,
        annotation_fp: str,
        make_partial: List[int] = None,
        fast_record_fp: str = None
    ):
        self.dataset_mean = COCO_MEAN
        self.dataset_std = COCO_STD
        if fast_record_fp is not None:
            fast_record_fp = fast_record_fp.format(split=self.split)
            try:
                self.logger.info("Found fast record file")
                record_dict: Dict[str, Any] = load_object(fast_record_fp)
                if not self._check_record(record_dict):
                    self.logger.warning("Record file `make_partial` incorrect, ignoring...")
                else:
                    self.CLASSES = record_dict["CLASSES"]
                    self.cat_id = record_dict["cat_id"]
                    self.cat_class_map = record_dict["cat_class_map"]
                    self.label_map = record_dict["label_map"]
                    self.label_info = record_dict["label_info"]
                    self.images = record_dict["images"]
                    self.image_ids = record_dict["image_ids"]
                    self.annotations = record_dict["annotations"]
                    return
            except:
                self.logger.warning("Load fast record file failed")

        self.logger.info("Reading annotation file...")
        with open(annotation_fp) as f:
            data = json.load(f)
        self.logger.info("Reading annotation file done")
        # initial classes
        self.CLASSES = tuple(c["name"] for c in data["categories"])
        self.cat_id = tuple(c["id"] for c in data["categories"])
        self.cat_class_map = {cat: c for cat, c in zip(self.cat_id, self.CLASSES)}

        if make_partial is not None:
            make_partial.sort()
            self.CLASSES = tuple(self.CLASSES[c] for c in make_partial)
            self.cat_id = tuple(self.cat_id[c] for c in make_partial)
            self.logger.info("Partial COCO%s %s dataset with classes: %s", self.version, self.split, str(self.CLASSES))

        # skip 0 for background
        for cls_id, cat in enumerate(self.CLASSES):
            cls_id = cls_id + 1
            self.label_map[cat] = cls_id
            self.label_info[cls_id] = cat

        # build inference for images
        self.images = [os.path.join(self.image_folder, x["file_name"]) for x in data["images"]]
        self.image_ids: List[int] = [x["id"] for x in data["images"]]
        self.annotations: Dict[int, DefaultDict[str, List]] = dict()
        for img_id in self.image_ids:
            self.annotations[img_id] = collections.defaultdict(list)

        # read annotations
        self.logger.info("Loading annotations...")
        for bbox in tqdm.tqdm(data["annotations"], desc="Loading Annotations"):
            img_id = bbox["image_id"]
            category_id = bbox["category_id"]
            class_name = self.cat_class_map[category_id]
            # class name does not count
            if class_name not in self.CLASSES:
                continue
            bbox_label = self.label_map[class_name]
            # 3 keys
            self.annotations[img_id]["boxes"].append(bbox["bbox"])
            self.annotations[img_id]["labels"].append(bbox_label)
            self.annotations[img_id]["iscrowd"].append(bbox["iscrowd"])

        assert len(self.annotations) == len(self.images)

        if fast_record_fp is not None:
            record_dict = {
                "CLASSES": self.CLASSES,
                "cat_id": self.cat_id,
                "cat_class_map": self.cat_class_map,
                "label_map": self.label_map,
                "label_info": self.label_info,
                "images": self.images,
                "image_ids": self.image_ids,
                "annotations": self.annotations,
                "make_partial": make_partial,
                "version": self.version,
                "split": self.split,
                "keep_crowd": self.keep_crowd,
            }
            # save only as main process
            if dist_utils.is_main_process():
                save_object(record_dict, fast_record_fp)
            dist_utils.barrier()

    def get_img_id(self, index: int) -> int:
        return self.image_ids[index]

    def get_annotation(self, index: int) -> Dict[str, Any]:
        img_id = self.image_ids[index]
        # transform to tensor
        # boxes with form [x, y, w, h]
        boxes = torch.tensor(self.annotations[img_id]["boxes"], dtype=torch.float).reshape(-1, 4)
        labels = torch.tensor(self.annotations[img_id]["labels"], dtype=torch.long)
        iscrowd = torch.tensor(self.annotations[img_id]["iscrowd"], dtype=torch.bool)
        if not self.keep_crowd:
            boxes = boxes[~iscrowd]
            labels = labels[~iscrowd]
            iscrowd = iscrowd[~iscrowd]
        target = {
            "boxes": boxes,
            "labels": labels,
            "iscrowd": iscrowd
        }
        return target
