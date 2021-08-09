import random
from collections import namedtuple
from typing import List

import torch
import numpy as np


LangMeta = namedtuple("LangMeta", ["lang_id", "lang_code", "max_length"])


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


def checkpoint_stats2string(step, avg_stats_dict, mode="train"):
    prt_msg = "step {:5d} {} ".format(step, mode)
    for name, value in avg_stats_dict.items():
        prt_msg += f"| {name}"
        prt_msg += " {:.4f}".format(value)
    prt_msg += " |"
    return prt_msg


def translation2string(translation_dict, num_printed_translation):
    prt_msg = ""
    for lang_id, translations in translation_dict.items():
        translations: List[str]
        prt_msg += f"Target Language ID: {lang_id} \n"
        num_printed_translation = min(num_printed_translation, len(translations))
        random_indices = np.random.choice(len(translations), num_printed_translation)
        for i, idx in enumerate(random_indices):
            source, translated = translations[idx]
            prt_msg += f"source {i}: {source}\n"
            prt_msg += f"translated {i}: {translated}\n"
            prt_msg += "\n"
    return prt_msg


def _lr_lambda(
    total_num_steps,
    num_warmup_steps=0,
    warmup='linear',
    decay='linear',
    decay_end_percent=0.0,
    gamma=0.9,
    gamma_steps=1000
):

    if warmup == 'flat':
        warmup_lambda = lambda step: 1.0
    elif warmup == 'linear':
        warmup_lambda = lambda step: (step + 1) / (num_warmup_steps)
    else:
        raise ValueError(f'Warmup mode {warmup} is not valid')

    if decay == 'linear':
        total_decay_steps = total_num_steps - num_warmup_steps
        percent_to_decay = 1.0 - decay_end_percent
        decay_lambda = lambda step: (
            (
                percent_to_decay *
                (total_num_steps - (step + 1)) / total_decay_steps
            ) + decay_end_percent
        )
    elif decay == 'exponential':
        decay_lambda = lambda step: (
            gamma**((step + 1 - num_warmup_steps) / gamma_steps)
        )
    elif decay == 'none':
        decay_lambda = lambda step: 1.0
    else:
        raise ValueError(f'Decay mode {decay} is not valid')

    lr_lambda = lambda step: (
        warmup_lambda(step) if step < num_warmup_steps else decay_lambda(step)
    )
    return lr_lambda


def get_lr_lambda_by_steps(
    total_num_steps,
    num_warmup_steps=0,
    warmup='linear',
    decay='linear',
    decay_end_percent=0.0,
    gamma=0.9,
    gamma_steps=1000
):
    """
    Get learning-rate step lambda based on the total nummber of steps and the
    number of warmup steps

    Warmup can be `flat` or `linear`. Decay can be `linear` or `exponential`.
    Exponential decay is defined by the `gamma` base and `gamma_steps` period
    over which the decay applies. For instance, if `gamma` is 0.5 and
    `gamma_steps` is 1000, the learning rate will decay by half every 1000 steps
    """

    return _lr_lambda(
        total_num_steps,
        num_warmup_steps=num_warmup_steps,
        warmup=warmup,
        decay=decay,
        decay_end_percent=decay_end_percent,
        gamma=gamma,
        gamma_steps=gamma_steps
    )
