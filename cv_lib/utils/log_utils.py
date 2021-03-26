import logging
from typing import Optional
import time
import shutil

from torch.utils.tensorboard import SummaryWriter

import cv_lib.distributed as dist_utils


__all__ = [
    "get_root_logger",
    "rm_tf_logger",
    "get_master_logger"
]


def get_root_logger(
    level: int,
    logger_fp: str,
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
