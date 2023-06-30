import os
import argparse
import random
from typing import Any, Dict, List
from collections import OrderedDict
import json
import pickle

import numpy as np

import torch
import torch.nn as nn
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
    "customized_argsort",
    "customized_sort",
    "tensor_dict_items",
    "tensor_to_list",
    "random_pick_instances",
    "check_nan_grad"
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


def count_parameters(model: nn.Module, include_no_grad: bool = False):
    if include_no_grad:
        return sum(p.numel() for p in model.state_dict().values())
    else:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


def to_json_str(obj: Any, indent: int = 4):
    return json.dumps(obj, allow_nan=True, indent=indent)


def float_to_uint_image(img: np.ndarray) -> np.ndarray:
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


def save_object(obj: Any, fp: str):
    fp = os.path.expanduser(fp)
    with open(fp, "wb") as f:
        return pickle.dump(obj, f)


def load_object(fp: str) -> Any:
    fp = os.path.expanduser(fp)
    with open(fp, "rb") as f:
        return pickle.load(f)


# will be replace by torch-1.9.0
def customized_argsort(tensor: torch.Tensor, dim=-1, descending=False, kind="quicksort"):
    """
    Only support tensor on cpu and without grad
    Args:
        kind: {'quicksort', 'mergesort', 'heapsort', 'stable'}
    """
    assert tensor.device == torch.device("cpu"), "Only support cpu tensor"
    assert tensor.requires_grad == False, "Only support tensor without grad"
    tensor_np = tensor.numpy()
    if descending:
        indices = np.argsort(-tensor_np, axis=dim, kind=kind)
    else:
        indices = np.argsort(tensor_np, axis=dim, kind=kind)
    return torch.from_numpy(indices).long()


# will be replace by torch-1.9.0
def customized_sort(tensor: torch.Tensor, dim=-1, descending=False, kind="quicksort"):
    """
    Only support tensor on cpu and without grad
    Args:
        kind: {'quicksort', 'mergesort', 'heapsort', 'stable'}
    """
    assert tensor.device == torch.device("cpu"), "Only support cpu tensor"
    assert tensor.requires_grad == False, "Only support tensor without grad"
    tensor_np = tensor.numpy()
    if descending:
        indices = np.sort(-tensor_np, axis=dim, kind=kind)
    else:
        indices = np.sort(tensor_np, axis=dim, kind=kind)
    return torch.from_numpy(indices).type_as(tensor)


def tensor_dict_items(tensor_dict: Dict[str, torch.Tensor], ndigits: int = 4) -> Dict[str, float]:
    out_dict = dict()
    for k, v in tensor_dict.items():
        out_dict[k] = round(v.item(), ndigits)
    return out_dict


def tensor_to_list(tensor: torch.Tensor, ndigits: int = 4) -> List[torch.Tensor]:
    tensor_list = list(round(t, ndigits) for t in tensor.tolist())
    return tensor_list


def random_pick_instances(instances: List[Any], make_partial: float, seed: int):
    if make_partial is None:
        return instances
    assert 0 < make_partial < 1
    n_pick = round(len(instances) * make_partial)
    rng = np.random.default_rng(seed=seed)
    rng.shuffle(instances)
    instances = instances[:n_pick]
    return instances


def check_nan_grad(model: nn.Module) -> List[str]:
    nan_list = []
    for k, v in model.named_parameters():
        if v.grad is not None and v.grad.isnan().any():
            nan_list.append(k)
    return nan_list
