import logging
from typing import Optional, Tuple
import time
import shutil
import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

import cv_lib.distributed as dist_utils


__all__ = [
    "get_root_logger",
    "rm_tf_logger",
    "get_master_logger",
    "get_tb_writer",
    "get_train_logger",
    "get_eval_logger",
]


def get_root_logger(
    level: int = logging.INFO,
    logger_fp: Optional[str] = None,
    name: Optional[str] = None,
    mode: str = "w",
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter(format)
    if logger_fp is not None:
        file_handler = logging.FileHandler(logger_fp, mode)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    logger.addHandler(console)
    return logger


def rm_tf_logger(writer: SummaryWriter):
    log_dir = writer.log_dir
    writer.close()
    shutil.rmtree(log_dir)
    time.sleep(1.5)


class DumpLogger:
    def setLevel(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def warn(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def exception(self, *args, exc_info=True, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass

    fatal = critical

    def log(self, *args, **kwargs):
        pass

    def makeRecord(self, *args, **kwargs):
        pass

    def _log(self, *args, **kwargs):
        pass

    def handle(self, record):
        pass


def get_master_logger(name: str = None):
    """
    Get logger only work on the master process
    """
    if dist_utils.is_main_process():
        return logging.getLogger(name)
    else:
        return DumpLogger()


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
