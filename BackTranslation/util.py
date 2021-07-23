import random

import numpy as np
from typing import List


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
        prt_msg += f"Language ID: {lang_id} \n"
        num_printed_translation = min(num_printed_translation, len(translations))
        random_indices = np.random.choice(len(translations), num_printed_translation)
        for i in random_indices:
            prt_msg += translations[i] + "\n"
        prt_msg += "\n"
    return prt_msg
