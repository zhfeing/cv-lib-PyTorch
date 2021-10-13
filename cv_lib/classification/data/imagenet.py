import os
from typing import Callable, Tuple, Optional, Dict, Any

from PIL import Image
import scipy.io as sio

import torch
from torchvision.datasets.utils import verify_str_arg
from torchvision.datasets.folder import pil_loader, make_dataset

from cv_lib.utils import log_utils
from cv_lib.utils import load_object, save_object
import cv_lib.distributed.utils as dist_utils


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
        fast_record_fp: str = None
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
        self.data_folder = os.path.join(self.root, f"{self.split}_set")
        self.meta_folder = os.path.join(self.root, "devkit", "data")

        self.logger = log_utils.get_master_logger("Imagenet")
        self._init_dataset(fast_record_fp)

    def _init_dataset(self, fast_record_fp: str):
        self.dataset_mean = MEAN
        self.dataset_std = STD
        if fast_record_fp is not None:
            fast_record_fp = fast_record_fp.format(split=self.split)
            try:
                self.logger.info("Found fast record file")
                record_dict: Dict[str, Any] = load_object(fast_record_fp)
                if not record_dict["split"] == self.split:
                    self.logger.warning("Record file `make_partial` incorrect, ignoring...")
                else:
                    self.CLASSES = record_dict["CLASSES"]
                    self.label_map = record_dict["label_map"]
                    self.label_info = record_dict["label_info"]
                    self.instances = record_dict["instances"]
                    return
            except:
                self.logger.warning("Load fast record file failed")

        self.logger.info("Reading annotation file...")
        # get classes and class map
        self.CLASSES = [d.name for d in os.scandir(self.data_folder) if d.is_dir()]
        self.CLASSES.sort()
        for i, cls_name in enumerate(self.CLASSES):
            self.label_info[i] = cls_name
            self.label_map[cls_name] = i
        self.instances = make_dataset(self.data_folder, self.label_map)

        if fast_record_fp is not None:
            record_dict = {
                "CLASSES": self.CLASSES,
                "label_map": self.label_map,
                "label_info": self.label_info,
                "instances": self.instances,
                "split": self.split,
            }
            # save only as main process
            if dist_utils.is_main_process():
                save_object(record_dict, fast_record_fp)

    def __len__(self):
        return len(self.images)

    def get_image(self, index: int) -> Image:
        image_fp = os.path.join(self.data_folder, self.instances[index][0])
        image = pil_loader(image_fp)
        return image

    def get_annotation(self, index: int) -> Dict[str, Any]:
        return self.instances[index][1]

    def _parse_meta_file(self):
        """
        Parse the devkit archive of the ImageNet2012 classification dataset and save
        the meta information in a binary file.
        """

        def parse_meta_mat() -> Tuple[Dict[int, str], Dict[str, str]]:
            metafile = os.path.join(self.meta_folder, "meta.mat")
            meta = sio.loadmat(metafile, squeeze_me=True)["synsets"]
            nums_children = list(zip(*meta))[4]
            meta = [meta[idx] for idx, num_children in enumerate(nums_children) if num_children == 0]
            idcs, wnids, classes = list(zip(*meta))[:3]
            classes = [tuple(clss.split(", ")) for clss in classes]
            idx_to_wnid = {idx: wnid for idx, wnid in zip(idcs, wnids)}
            wnid_to_classes = {wnid: clss for wnid, clss in zip(wnids, classes)}
            return idx_to_wnid, wnid_to_classes

        def parse_val_groundtruth_txt(devkit_root: str) -> List[int]:
            file = os.path.join(devkit_root, "data", "ILSVRC2012_validation_ground_truth.txt")
            with open(file, "r") as txtfh:
                val_idcs = txtfh.readlines()
            return [int(val_idx) for val_idx in val_idcs]

        idx_to_wnid, wnid_to_classes = parse_meta_mat()
        val_idcs = parse_val_groundtruth_txt(devkit_root)
        val_wnids = [idx_to_wnid[idx] for idx in val_idcs]

        torch.save((wnid_to_classes, val_wnids), os.path.join(root, META_FILE))

