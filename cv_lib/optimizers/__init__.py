from typing import Dict, Any, List
import copy
from functools import partial

from torch import nn
from torch.optim import *

from cv_lib.utils import to_json_str
from cv_lib.utils import log_utils
from .sam import SAM


__REGISTERED_OPTIMIZERS__: Dict[str, Optimizer] = {
    "SGD": SGD,
    "Adam": Adam,
    "AdamW": AdamW,
    "ASGD": ASGD,
    "Adamax": Adamax,
    "Adadelta": Adadelta,
    "Adagrad": Adagrad,
    "RMSprop": RMSprop,
}


def get_optimizer(params: List[Dict[str, nn.Parameter]], optimizer_cfg: Dict[str, Dict[str, Any]]) -> Optimizer:
    """
    optimizer yml example
    ```
        optimizer:
            name: sgd
            lr: 0.01
            weight_decay: 5.0e-4
            momentum: 0.9
    ```
    """
    logger = log_utils.get_master_logger("get_optimizer")

    opt_name = optimizer_cfg["name"]
    optim_cfg = copy.deepcopy(optimizer_cfg)
    optim_cfg.pop("name")

    logger.info("Using {} optimizer with config {}".format(opt_name, to_json_str(optim_cfg)))
    optimizer = __REGISTERED_OPTIMIZERS__[opt_name](params, **optim_cfg)
    return optimizer


def get_sam_optimizer(params: List[Dict[str, nn.Parameter]], optimizer_cfg: Dict[str, Dict[str, Any]]) -> SAM:
    logger = log_utils.get_master_logger("get_sam_optimizer")
    optimizer_cfg = copy.deepcopy(optimizer_cfg)
    base_cfg = optimizer_cfg["base"]
    sam_cfg = optimizer_cfg["sam"]
    base_name = base_cfg.pop("name")

    logger.info("Using {} base optimizer with base config {}, sam config {}".format(
        base_name,
        to_json_str(sam_cfg),
        to_json_str(sam_cfg),
    ))
    base_optimizer = partial(
        __REGISTERED_OPTIMIZERS__[base_name],
        **base_cfg
    )
    sam = SAM(
        params,
        base_optimizer=base_optimizer,
        **sam_cfg
    )
    return sam

