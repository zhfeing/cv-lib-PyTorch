import logging

from torch.utils.data import DataLoader


def make_iter_dataloader(data_loader: DataLoader):
    ep = 1
    logger = logging.getLogger("make_iter_dataloader")
    while True:
        for data in data_loader:
            yield data
        logger.info("Epoch %d finished, start to iter next epoch", ep)
        ep += 1

