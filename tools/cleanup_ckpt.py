import os
import argparse


def rm_ckpts(ckpt_dir: str, test: bool, perserve_first: bool):
    for dirpath, _, filenames in os.walk(ckpt_dir, followlinks=True):
        if "best.pth" in filenames:
            print("\nIn dir:", dirpath)
            filenames.sort()
            filenames.pop(0)
        if len(filenames) == 0:
            return
        assert "best.pth" not in filenames
        iters = list(int(a.split("-")[1][:-4]) for a in filenames)
        iters.sort()
        iters.pop(-1)
        if perserve_first:
            if len(iters) < 1:
                return
            iters.pop(0)
        for it in iters:
            fp = os.path.join(dirpath, "iter-{}.pth".format(it))
            if test:
                print(fp, os.path.isfile(fp))
            else:
                os.remove(fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--preserve-first-ckpt", action="store_true")
    args = parser.parse_args()

    root = os.path.expanduser(args.root)
    ckpt_dirs = list()
    for dirpath, dirnames, _ in os.walk(root):
        if "ckpt" in dirnames:
            ckpt_dir = os.path.join(dirpath, "ckpt")
            ckpt_dirs.append(ckpt_dir)

    for dir in ckpt_dirs:
        rm_ckpts(dir, args.test, args.preserve_first_ckpt)
