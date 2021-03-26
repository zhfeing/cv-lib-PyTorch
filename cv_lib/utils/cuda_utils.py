import os
import gc
import threading
from typing import List

import torch
from torch import Tensor

from cv_lib.utils import log_utils


__all__ = [
    "MemoryPreserveError",
    "all_tensors",
    "preserve_memory",
    "list_to_device"
]


class MemoryPreserveError(Exception):
    pass


def all_tensors():
    dtype_32 = [torch.float32, torch.int32]
    dtype_64 = [torch.float64, torch.int64]
    dtype_16 = [torch.float16, torch.int16]
    dtype_8 = [torch.uint8, torch.int8]
    # in MB
    total_mem = 0
    for obj in gc.get_objects():
        if isinstance(obj, torch.Tensor):
            mem = obj.numel()
            if obj.dtype in dtype_8:
                pass
            elif obj.dtype in dtype_16:
                mem *= 2
            elif obj.dtype in dtype_32:
                mem *= 4
            elif obj.dtype in dtype_64:
                mem *= 8
            else:
                print("dtype: {} unknown, set to dtype_32")
                mem *= 4
            total_mem += mem
    print("total cached mem: {:.3f}MB".format(total_mem / 1024 / 1024))


def preserve_gpu_with_id(gpu_id: int, preserve_percent: float = 0.95):
    logger = log_utils.get_master_logger("preserve_gpu_with_id")
    if not torch.cuda.is_available():
        logger.warning("no gpu avaliable exit...")
        return
    try:
        import cupy
        device = cupy.cuda.Device(gpu_id)
        avaliable_mem = device.mem_info[0] - 700 * 1024 * 1024
        logger.info("{}MB memory avaliable, trying to preserve {}MB...".format(
            int(avaliable_mem / 1024.0 / 1024.0),
            int(avaliable_mem / 1024.0 / 1024.0 * preserve_percent)
        ))
        if avaliable_mem / 1024.0 / 1024.0 < 100:
            cmd = os.popen("nvidia-smi")
            outputs = cmd.read()
            pid = os.getpid()

            logger.warning("Avaliable memory is less than 100MB, skiping...")
            logger.info("program pid: %d, current environment:\n%s", pid, outputs)
            raise MemoryPreserveError()
        alloc_mem = int(avaliable_mem * preserve_percent / 4)
        x = torch.empty(alloc_mem).to(torch.device("cuda:{}".format(gpu_id)))
        del x
    except ImportError:
        logger.warning("No cupy found, memory cannot be perserved")


def preserve_memory(preserve_percent: float = 0.99):
    logger = log_utils.get_master_logger("preserve_memory")
    if not torch.cuda.is_available():
        logger.warning("no gpu avaliable exit...")
        return
    thread_pool = list()
    for i in range(torch.cuda.device_count()):
        thread = threading.Thread(
            target=preserve_gpu_with_id,
            kwargs=dict(
                gpu_id=i,
                preserve_percent=preserve_percent
            ),
            name="Preserving GPU {}".format(i)
        )
        logger.info("Starting to preserve GPU: {}".format(i))
        thread.start()
        thread_pool.append(thread)
    for t in thread_pool:
        t.join()


def list_to_device(src: List[Tensor], device: torch.device) -> List[Tensor]:
    dst = list(x.to(device) for x in src)
    return dst

