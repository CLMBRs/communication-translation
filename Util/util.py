import logging
import sys
import random
import torch
import numpy as np


def create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    out_handler = logging.StreamHandler()
    out_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        )
    )
    out_handler.setLevel(logging.INFO)
    logger.addHandler(out_handler)
    return logger


def set_seed(seed, n_gpu=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(seed)


def statbar_string(stat_dict: dict) -> str:
    """
    Return a printable "statbar" string from a dictionary of named statistics
    """
    stat_items = []
    for key, value in stat_dict.items():
        stat_items.append(f"{key} {value}")
    return ' | '.join(stat_items)
