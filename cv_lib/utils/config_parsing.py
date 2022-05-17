import yaml
from typing import Dict, Any


def get_cfg(cfg_filepath: str) -> Dict[str, Any]:
    with open(cfg_filepath) as fp:
        global_cfg = yaml.load(fp, Loader=yaml.SafeLoader)
        return global_cfg


def set_debug(global_cfg: Dict[str, Any], debug: bool = False):
    if debug:
        global_cfg["training"]["n_workers"] = 0
        global_cfg["validation"]["n_workers"] = 0
