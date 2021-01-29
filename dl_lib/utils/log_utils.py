import logging
from typing import Optional
import time
import shutil

from torch.utils.tensorboard import SummaryWriter


__all__ = [
    "get_logger",
    "rm_tf_logger"
]


def get_logger(
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

