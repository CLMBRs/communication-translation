import logging
import sys
import random
import torch
import numpy as np


def create_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setFormatter(
        logging.Formatter(
            fmt='%(asctime)s - %(message)s', datefmt='%m-%d-%y %H:%M:%S'
        )
    )
    out_handler.setLevel(logging.INFO)
    logger.addHandler(out_handler)
    return logger


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def statbar_string(stat_dict: dict) -> str:
    """
    Return a printable "statbar" string from a dictionary of named statistics
    """
    stat_items = []
    for key, value in stat_dict.items():
        stat_items.append(f"{key} {value}")
    return ' | '.join(stat_items)
