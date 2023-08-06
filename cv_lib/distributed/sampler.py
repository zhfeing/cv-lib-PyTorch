import torch
import torch.utils.data as data


def get_train_sampler(
    distributed: bool,
    train_dataset: data.Dataset,
) -> data.Sampler:
    if distributed:
        train_sampler = data.DistributedSampler(train_dataset, shuffle=True)
    else:
        generator = torch.Generator()
        generator.manual_seed(0)
        train_sampler = data.RandomSampler(train_dataset, generator=generator)
    return train_sampler


def get_val_sampler(
    distributed: bool,
    val_dataset: data.Dataset
) -> data.Sampler:
    if distributed:
        val_sampler = data.DistributedSampler(val_dataset, shuffle=False)
    else:
        val_sampler = data.SequentialSampler(val_dataset)
    return val_sampler

