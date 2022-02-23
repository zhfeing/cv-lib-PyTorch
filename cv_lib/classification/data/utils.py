import os
import collections
from typing import OrderedDict, Dict, List, Tuple

from torchvision.datasets.folder import is_image_file

from cv_lib.utils import random_pick_instances


def make_datafolder(data_folder: str, make_partial: float = None, manual_classes: List[str] = None):
    # get classes and class map
    if manual_classes:
        classes = manual_classes
    else:
        classes = [d.name for d in os.scandir(data_folder) if d.is_dir()]
        classes.sort()
    label_map: OrderedDict[str, int] = collections.OrderedDict()
    label_info: OrderedDict[int, str] = collections.OrderedDict()
    for i, cls_name in enumerate(classes):
        label_info[i] = cls_name
        label_map[cls_name] = i
    instances_by_class = make_dataset(data_folder, label_map)
    instances: List[Tuple[str, int]] = []
    for i, ins in enumerate(instances_by_class):
        ins = random_pick_instances(ins, make_partial, seed=i)
        instances.extend(ins)
    return instances, label_info, label_map


def make_dataset(
    directory: str,
    class_to_idx: Dict[str, int],
) -> List[List[Tuple[str, int]]]:
    """
    Generates a list of samples of a form (path_to_sample, class).
    Args:
        directory (str): root dataset directory
        class_to_idx (Dict[str, int]): dictionary mapping class name to class index
    Returns:
        List[List[Tuple[str, int]]]: samples of a form (path_to_sample, class) by class_id
    """
    instances_by_class: List[List[Tuple[str, int]]] = []
    directory = os.path.expanduser(directory)

    for target_class in sorted(class_to_idx.keys()):
        instances: List[Tuple[str, int]] = []
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            raise Exception("Class {} has no image files".format(target_class))
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_image_file(path):
                    item = path, class_index
                    instances.append(item)
            instances_by_class.append(instances)
    return instances_by_class

