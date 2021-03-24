import os
from datetime import datetime
import yaml
import logging
from typing import Dict, Any, Tuple

from torch.utils.tensorboard import SummaryWriter

from .utils import get_root_logger


def get_cfg(cfg_filepath: str) -> Dict[str, Any]:
    with open(cfg_filepath) as fp:
        global_cfg = yaml.load(fp, Loader=yaml.SafeLoader)
        return global_cfg


def set_debug(global_cfg: Dict[str, Any], debug: bool = False):
    if debug:
        global_cfg["training"]["n_workers"] = 0
        global_cfg["validation"]["n_workers"] = 0


def get_tb_writer(logdir, filename) -> Tuple[SummaryWriter, str]:
    logger_fp = os.path.join(
        logdir,
        "tf-board-logs",
        filename
    )
    writer = SummaryWriter(
        logger_fp,
        flush_secs=1,
        filename_suffix="_datetime-{}_".format(datetime.now())
    )
    return writer, logger_fp


def get_train_logger(logdir: str, filename: str, mode="w") -> Tuple[logging.Logger, str]:
    train_log_dir = os.path.join(logdir, "train-logs")
    os.makedirs(train_log_dir, exist_ok=True)
    logger_fp = os.path.join(
        train_log_dir,
        "training-" + filename + ".log"
    )
    logger = get_root_logger(
        level=logging.INFO,
        mode=mode,
        name=None,
        logger_fp=logger_fp
    )
    logger.propagate = False
    return logger, logger_fp


def get_eval_logger(logdir: str) -> Tuple[logging.Logger, str]:
    os.makedirs(logdir, exist_ok=True)
    logger_fp = os.path.join(
        logdir,
        "eval.log"
    )
    logger = get_root_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=logger_fp
    )
    return logger, logger_fp
