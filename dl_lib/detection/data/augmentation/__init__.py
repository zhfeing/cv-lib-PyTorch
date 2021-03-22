from typing import List, Dict, Any

from .aug import *
from dl_lib.utils import log_utils


__REGISTERED_AUGS__ = {
    "RandomCrop": RandomCrop,
    "RandomSizeCrop": RandomSizeCrop,
    "CenterCrop": CenterCrop,
    "RandomHorizontalFlip": RandomHorizontalFlip,
    "ColorJitter": ColorJitter,
    "RandomPadBottomRight": RandomPadBottomRight,
    "RandomResize": RandomResize,
}


def get_composed_augmentations(aug_list: List[Dict[str, Any]]):
    logger = log_utils.get_master_logger("get_composed_augmentations")
    augmentations = list()
    for aug in aug_list:
        assert len(aug) == 1, "one aug each time"
        name, args = aug.popitem()
        aug_method = __REGISTERED_AUGS__[name]
        if isinstance(args, (list, tuple)):
            augmentations.append(aug_method(*args))
        elif isinstance(args, dict):
            augmentations.append(aug_method(**args))
        else:
            augmentations.append(aug_method(args))
        logger.info("Using {} aug with params {}".format(name, args))

    return Compose(*augmentations)
