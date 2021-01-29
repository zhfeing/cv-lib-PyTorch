import os
from datetime import datetime
import yaml
import logging
from typing import Dict, Any

from torch.utils.tensorboard import SummaryWriter

from basic.utils import get_logger


def get_cfg(cfg_filepath: str) -> Dict[str, Any]:
    with open(cfg_filepath) as fp:
        global_cfg = yaml.load(fp, Loader=yaml.SafeLoader)
        return global_cfg


def set_debug(global_cfg: Dict[str, Any], debug: bool = False):
    if debug:
        global_cfg["training"]["n_workers"] = 0
        global_cfg["validation"]["n_workers"] = 0


def get_tb_writer(logdir, filename) -> SummaryWriter:
    writer = SummaryWriter(
        log_dir=os.path.join(
            logdir,
            "tf-board-logs",
            filename
        ),
        flush_secs=1,
        filename_suffix="_datetime-{}_".format(datetime.now())
    )
    return writer


def get_train_logger(logdir: str, filename: str) -> logging.Logger:
    train_log_dir = os.path.join(logdir, "train-logs")
    os.makedirs(train_log_dir, exist_ok=True)
    logger = get_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=os.path.join(
            train_log_dir,
            "training-" + filename + ".log"
        )
    )
    return logger


def get_eval_logger(logdir: str) -> logging.Logger:
    os.makedirs(logdir, exist_ok=True)
    logger = get_logger(
        level=logging.INFO,
        mode="w",
        name=None,
        logger_fp=os.path.join(
            logdir,
            "eval.log"
        )
    )
    return logger
