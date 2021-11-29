import argparse
import os
from typing import Any, Dict, List, Tuple
import multiprocessing as mp

import tqdm
import pandas as pd
import numpy as np
import scipy.interpolate

from tensorboard.backend.event_processing import event_accumulator


def interpolation(df: pd.DataFrame, total_epoch: int) -> pd.DataFrame:
    step = df["step"].values
    epoch = step / step[-1] * total_epoch
    epoch_new = np.arange(1, total_epoch + 1)

    new_df = pd.DataFrame()
    new_df["step"] = epoch_new
    for column in df.columns:
        if column != "step":
            func = scipy.interpolate.UnivariateSpline(epoch, df[column], s=0, k=3)
            new_df[column] = func(epoch_new)
    return new_df


def process_events(
    base_dir: str,
    pre_fix: str,
    total_epoch: int,
    save_path: str,
    tqdm_queue: mp.Queue
) -> Dict[str, pd.DataFrame]:
    ea = event_accumulator.EventAccumulator(base_dir)
    ea.Reload()
    keys = ea.scalars.Keys()
    data_dict: Dict[str, Any] = dict()
    for key in keys:
        if pre_fix != "":
            new_key = f"{pre_fix}-{key}"
        else:
            new_key = key
        new_key = new_key.replace("/", "-")
        scalers = ea.Scalars(key)
        df = pd.DataFrame(scalers)
        start_time = df.loc[0, "wall_time"]
        df["relative_time"] = df.apply(lambda x: x["wall_time"] - start_time, axis=1)
        # remove duplicated
        mask = ~df["step"].duplicated(keep="first")
        df = df[mask]
        if total_epoch != -1:
            df = interpolation(df, total_epoch)
        data_dict[new_key] = df
        save_fp = os.path.join(save_path, f"{new_key}.csv")
        df.to_csv(save_fp)
    tqdm_queue.put(pre_fix)

    return data_dict


def tqdm_func(in_queue: mp.Queue, total: int):
    tqdm_bar = tqdm.tqdm(total=total)
    while True:
        val = in_queue.get()
        if val is None:
            break
        tqdm_bar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--write-path", type=str)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--total-epoch", type=int, default=-1)
    parser.add_argument("--filters", type=str, nargs="+", default=list())
    args = parser.parse_args()

    # preparation
    root = os.path.expanduser(args.root)
    os.makedirs(args.write_path, exist_ok=True)

    event_files: List[Tuple[str, str]] = list()
    m = mp.Manager()
    queue = m.Queue()

    sub_dirs = os.listdir(root)
    sub_dirs = [""] + sub_dirs
    task_args: List[Tuple[str, str]] = list()

    def filter(base_dir: str, sub_dir: str, filter_prefix: List[str]) -> bool:
        val = os.path.isdir(base_dir)
        match = False
        for prefix in filter_prefix:
            match = match or (sub_dir.find(prefix) != -1)
            if match:
                break
        return val and not match

    for dir in sub_dirs:
        base_dir = os.path.join(root, dir)
        if filter(base_dir, dir, args.filters):
            task_args.append((base_dir, dir, args.total_epoch, args.write_path, queue))

    tqdm_process = mp.Process(target=tqdm_func, args=(queue, len(task_args)))
    tqdm_process.start()

    with mp.Pool(args.num_workers) as p:
        results = p.starmap(process_events, task_args)
    queue.put(None)
    tqdm_process.join()


