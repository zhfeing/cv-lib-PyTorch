from typing import Any, Callable, Dict, List

import torch
from torch import Tensor
import torch.distributed as dist


__all__ = [
    "is_dist_avail_and_initialized",
    "get_world_size",
    "get_rank",
    "is_main_process",
    "all_gather",
    "all_gather_tensor",
    "all_gather_object",
    "all_gather_list",
    "reduce_tensor",
    "reduce_dict",
    "cal_split_args",
    "barrier",
    "run_on_main_process",
    "broadcast_tensor"
]


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def all_gather(data: Any, device: torch.device) -> List[Any]:
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    if isinstance(data, Tensor):
        return all_gather_tensor(data, device)
    else:
        return all_gather_object(data, device)


def all_gather_object(data: Any) -> List[Any]:
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)

    Note:
        For NCCL-based processed groups, internal tensor representations of objects
        must be moved to the GPU device before communication takes place. In this case,
        the device used is given by torch.cuda.current_device() and it is the user’s
        responsibility to ensure that this is set so that each rank has an individual
        GPU, via torch.cuda.set_device().

    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]

    data_list = [None] * get_world_size()
    dist.all_gather_object(data_list, data)
    return data_list


def all_gather_list(items: List[Any]) -> List[Any]:
    if get_world_size() == 1:
        return items

    item_list = [None] * get_world_size()
    dist.all_gather_object(item_list, items)
    ret = list()
    for items in item_list:
        ret.extend(items)
    return ret


def all_gather_tensor(data: Tensor, device: torch.device) -> List[Tensor]:
    """
    Run all_gather on Tensor

    Args:
        data: any Tensor
    Returns:
        list[Tensor]: list of tensor gathered from each rank
    """
    data = data.to(device)
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    data_list = list(torch.empty_like(data, device=device) for _ in range(world_size))
    dist.all_gather(data_list, data)
    return data_list


def reduce_tensor(tensor: torch.Tensor, average=True) -> Tensor:
    """
    Reduce torch.Tensor to sum or average from process group
    """
    world_size = get_world_size()
    if world_size < 2:
        return tensor
    with torch.no_grad():
        device = tensor.device
        tensor = tensor.to(torch.device("cuda:{}".format(get_rank())))
        dist.all_reduce(tensor)
        if average:
            tensor /= world_size
        return tensor.to(device)


def reduce_dict(input_dict: Dict[str, Tensor], average=True) -> Dict[str, Tensor]:
    """
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.

    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v for k, v in zip(names, values)}
    return reduced_dict


def cal_split_args(batch_size: int, n_workers: int, ngpus_per_node: int):
    """
    Calculate batch size and number of workers when distributed training,
    For each process, it should be smaller than total configs
    """
    batch_size = int(batch_size / ngpus_per_node)
    n_workers = int((n_workers + ngpus_per_node - 1) / ngpus_per_node)
    return batch_size, n_workers


def barrier():
    if get_world_size() > 1:
        dist.barrier()


def run_on_main_process(func: Callable, *args, **kwargs):
    if is_main_process():
        func(*args, **kwargs)
    barrier()


def broadcast_tensor(tensor: Tensor, src: int):
    if get_world_size() > 1:
        dist.broadcast(
            tensor=tensor,
            src=src
        )

