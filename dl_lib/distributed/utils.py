import pickle
from typing import Any, Dict, List

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
    "reduce_tensor",
    "reduce_dict",
    "cal_split_args",
    "barrier",
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


def all_gather_object(data: Any, device: torch.device) -> List[Any]:
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)

    Warnings:
        Do not use pytorch official gather_all_object which has bug when device is
        not manually specified
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    if get_world_size() == 1:
        return [data]

    # serialized to a Tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=device)
    size_list = list(torch.tensor([0], device=device) for _ in range(get_world_size()))
    dist.all_gather(size_list, local_size)
    size_list = list(int(size.item()) for size in size_list)
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list: List[Tensor] = list()
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device=device))
    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device=device)
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = list()
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


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
        dist.all_reduce(tensor)
        if average:
            tensor /= world_size
        return tensor


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

