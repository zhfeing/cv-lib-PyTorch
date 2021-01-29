import os
from typing import Callable, Dict, Tuple, List, Any
from collections import defaultdict

import numpy as np

import torch
import torchvision.transforms.functional as TF

from data.detection_dataset import DetectionDataset


class FewExampleDataset(DetectionDataset):
    def __init__(
        self,
        full_dataset: DetectionDataset,
        num_per_class: int = 10,
        seed: float = None
    ):
        self.full_dataset = full_dataset
        self.num_per_class = num_per_class
        # init from full dataset
        self.resize = full_dataset.resize
        self.augmentations = full_dataset.augmentations
        self.img_sub_folder = full_dataset.img_sub_folder
        self.label_map = full_dataset.label_map
        self.label_info = full_dataset.label_info
        self.dataset_mean = full_dataset.dataset_mean
        self.dataset_std = full_dataset.dataset_std

        images = full_dataset.images
        img_keys = full_dataset.img_keys

        self.rng = self._init_rng(seed)
        self._init_few_example(images, img_keys)

    @staticmethod
    def _init_rng(seed: float = None):
        if seed is None:
            seed = np.random.randint(1e10)
        return np.random.default_rng(seed)

    def _init_few_example(self, images: dict, img_keys: list):
        # shuffle images
        self.rng.shuffle(img_keys)
        img_list_by_class = {cls_id: [] for cls_id in self.label_map.values()}
        self.images = dict()

        def check_finish():
            finish = True
            for v in img_list_by_class.values():
                if len(v) < self.num_per_class:
                    finish = False
                    break
            return finish

        for idx in img_keys:
            img_file, bboxes = images[idx]
            num_annots = len(bboxes)
            # remove complex img
            if num_annots > 5 or num_annots == 0:
                continue
            other_info = self.full_dataset.other_info(idx)
            difficult = other_info.get("difficult", [False] * num_annots)
            for i, (bbox, label) in enumerate(bboxes):
                # insert a few shot img
                if len(img_list_by_class[label]) < self.num_per_class and not difficult[i]:
                    img_list_by_class[label].append(idx)
                    self.images[idx] = (img_file, [(bbox, label)])
                    break
            if check_finish():
                break

        self.img_keys = list()
        for v in img_list_by_class.values():
            self.img_keys.extend(v)

