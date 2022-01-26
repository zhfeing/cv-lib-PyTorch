import argparse
import os
import shutil

import tqdm
import pandas as pd


def create_folder(
    root: str,
    dest_path: str,
    split_data: pd.DataFrame
):
    for _, row in tqdm.tqdm(split_data.iterrows(), total=split_data.shape[0]):
        path = row["directory"]
        name = row["img_name"]
        cp_path = os.path.join(dest_path, path)
        os.makedirs(cp_path, exist_ok=True)
        img_fp = os.path.join(root, path, name)
        dest_fp = os.path.join(cp_path, name)
        shutil.copyfile(img_fp, dest_fp, follow_symlinks=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str)
    parser.add_argument("--dest-path", type=str)
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    dest_path = os.path.expanduser(args.dest_path)
    train_split = os.path.join(root, "train_metadata.csv")
    val_split = os.path.join(root, "val_metadata.csv")

    with open(train_split, "r") as f:
        train_split = pd.read_csv(f)
    with open(val_split, "r") as f:
        val_split = pd.read_csv(f)

    create_folder(root, os.path.join(dest_path, "train"), train_split)
    create_folder(root, os.path.join(dest_path, "val"), val_split)
