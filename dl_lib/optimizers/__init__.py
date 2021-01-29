import logging
from typing import Dict, Any

from torch.optim import SGD, Adam, ASGD, Adamax, Adadelta, Adagrad, RMSprop


__REGISTERED_OPTIMIZERS__ = {
    "SGD": SGD,
    "Adam": Adam,
    "ASGD": ASGD,
    "Adamax": Adamax,
    "Adadelta": Adadelta,
    "Adagrad": Adagrad,
    "RMSprop": RMSprop,
}


def get_optimizer(optimizer_cfg: Dict[str, Dict[str, Any]]):
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
    logger = logging.getLogger("get_optimizer")

    opt_name = optimizer_cfg["name"]
    if opt_name not in __REGISTERED_OPTIMIZERS__:
        raise NotImplementedError("Optimizer {} not implemented".format(opt_name))

    logger.info("Using {} optimizer".format(opt_name))
    return __REGISTERED_OPTIMIZERS__[opt_name]
