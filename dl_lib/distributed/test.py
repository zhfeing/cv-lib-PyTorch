import argparse
import time

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from models.resnetv2 import ResNet101
from dataset.cifar import get_cifar_100
from helper.util import make_deterministic
# from torch.optim.lr_scheduler import MultiStepLR


__global_values__ = {
    "acc": None,
    "batch_size": 32,
    "n_workers": 8
}


def main():
    parser = argparse.ArgumentParser(description="PyTorch ImageNet Training")
    parser.add_argument("--num-nodes", default=-1, type=int, help="number of nodes for distributed training")
    parser.add_argument("--rank", default=-1, type=int, help="node rank for distributed training")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:9876", type=str)
    parser.add_argument("--dist-backend", default="nccl", type=str, help="distributed backend")
    parser.add_argument("--seed", default=None, type=int, help="seed for initializing training.")
    parser.add_argument("--multiprocessing", action="store_true")
    parser.add_argument("--file-name-cfg", type=str)
    parser.add_argument("--log-dir", type=str)
    args = parser.parse_args()

    if args.seed is not None:
        print("Set seed:", args.seed)
        make_deterministic(args.seed)
    torch.backends.cudnn.benchmark = True

    runner = Runner(
        num_nodes=args.num_nodes,
        rank=args.rank,
        dist_url=args.dist_url,
        dist_backend=args.dist_backend,
        multiprocessing=args.multiprocessing
    )
    print("Starting runner")
    runner()


class Runner:
    def __init__(
        self,
        num_nodes: int,
        rank: int,
        dist_url: str,
        dist_backend: str,
        multiprocessing: bool
    ):
        # if more than one node or using multiprocess for each node
        self.num_nodes = num_nodes
        self.rank = rank
        self.dist_url = dist_url
        self.dist_backend = dist_backend
        self.multiprocessing = multiprocessing
        self.ngpus_per_node = torch.cuda.device_count()

        self.world_size = self.num_nodes
        if self.multiprocessing:
            self.world_size = self.ngpus_per_node * self.num_nodes

        self.distributed = self.world_size > 1
        print("Distributed training:", self.distributed)

    def __call__(self):
        if self.multiprocessing:
            print("Start from multiprocessing")
            mp.spawn(self.worker, nprocs=self.ngpus_per_node)
        else:
            print("Start from direct call")
            self.worker(0)

    def worker(self, gpu_id: int):
        pass
        # print("Use GPU: {} for training".format(gpu_id))
        # current_rank = self.rank
        # if self.distributed:
        #     if self.multiprocessing:
        #         # For multiprocessing distributed training, rank needs to be the
        #         # global rank among all the processes
        #         current_rank = self.rank * self.ngpus_per_node + gpu_id
        #     print("Current rank:", current_rank)
        #     dist.init_process_group(
        #         backend=self.dist_backend,
        #         init_method=self.dist_url,
        #         world_size=self.world_size,
        #         rank=current_rank
        #     )
        # # create model
        # model = ResNet101(num_classes=100)

        # device = torch.device("cuda:{}".format(gpu_id))
        # model.to(device)

        # batch_size = __global_values__["batch_size"]
        # n_workers = __global_values__["n_workers"]
        # if self.distributed:
        #     batch_size = int(__global_values__["batch_size"] / self.ngpus_per_node)
        #     n_workers = int((__global_values__["n_workers"] + self.ngpus_per_node - 1) / self.ngpus_per_node)
        #     model = DistributedDataParallel(model, device_ids=[gpu_id])
        # print("batch_size: {}, workers: {}".format(batch_size, n_workers))

        # # define loss function (criterion) and optimizer
        # criterion = nn.CrossEntropyLoss().to(device)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9, weight_decay=1e-4)
        # # scheduler = MultiStepLR(optimizer, [10, 20], 0.1)
        # train_set = get_cifar_100("~/datasets/cifar")
        # val_set = get_cifar_100("~/datasets/cifar", "test")

        # if self.distributed:
        #     train_sampler = DistributedSampler(train_set)
        # else:
        #     train_sampler = None

        # train_loader = DataLoader(
        #     train_set,
        #     batch_size=batch_size,
        #     shuffle=(train_sampler is None),
        #     num_workers=n_workers,
        #     pin_memory=True,
        #     sampler=train_sampler
        # )

        # val_loader = DataLoader(
        #     val_set,
        #     batch_size=batch_size,
        #     shuffle=False,
        #     num_workers=n_workers,
        #     pin_memory=True
        # )

        # for epoch in range(10):
        #     if self.distributed:
        #         train_sampler.set_epoch(epoch)
        #     # train for one epoch
        #     train(train_loader, model, criterion, optimizer, device)
        #     # evaluate on validation set
        #     acc1 = validate(val_loader, model, criterion)
        #     print("acc@1:", acc1)


if __name__ == "__main__":
    main()

