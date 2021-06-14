from typing import Tuple

from torch.utils.data import Sampler, Dataset, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler


def get_train_sampler(
    distributed: bool,
    train_dataset: Dataset
) -> Sampler:
    if distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)
    return train_sampler


def get_val_sampler(
    distributed: bool,
    val_dataset: Dataset
) -> Sampler:
    if distributed:
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = SequentialSampler(val_dataset)
    return val_sampler

