import os
import argparse
import random
from typing import Any
from collections import OrderedDict
import json
import pickle
import logging

import numpy as np


from torch.utils.data import DataLoader
import torch
from torchvision.datasets.folder import IMG_EXTENSIONS
from torchvision.datasets.folder import has_file_allowed_extension


__all__ = [
    "is_valid_file",
    "recursive_glob",
    "convert_state_dict",
    "make_deterministic",
    "str2bool",
    "count_parameters",
    "to_json_str",
    "float_to_uint_image",
    "save_object",
    "load_object",
]


def is_valid_file(x):
    return has_file_allowed_extension(x, IMG_EXTENSIONS)


def recursive_glob(rootdir="."):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    file_list = [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if is_valid_file(filename)
    ]
    file_list.sort()
    return file_list


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    if not next(iter(state_dict)).startswith("module."):
        # abort if dict is not a DataParallel model_state
        return state_dict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        # remove `module.`
        name = k[7:]
        new_state_dict[name] = v
    return new_state_dict


def make_deterministic(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.set_deterministic(True)
    torch.backends.cudnn.deterministic = True


def str2bool(v):
    if v.lower() in ("true", "yes", "t", "y"):
        return True
    elif v.lower() in ("false", "no", "f", "n"):
        return False
    else:
        raise argparse.ArgumentTypeError("Unsupported value encountered.")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_json_str(obj: Any, indent: int = 4):
    return json.dumps(obj, allow_nan=True, indent=indent)


def float_to_uint_image(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def save_object(obj: Any, fp: str):
    with open(fp, "wb") as f:
        return pickle.dump(obj, f)


def load_object(fp: str) -> Any:
    with open(fp, "rb") as f:
        return pickle.load(f)
