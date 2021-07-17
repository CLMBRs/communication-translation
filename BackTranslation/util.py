import random

import numpy as np
from typing import List

def checkpoint_stats2string(step, avg_loss_dict, mode="train"):
    prt_msg = "step {:5d} {} ".format(step, mode)
    prt_msg += "| loss"
    prt_msg += " {:.4f}".format(avg_loss_dict["loss"])
    # prt_msg += "| prediction accuracy"
    # prt_msg += " {:.2f}%".format(avg_loss_dict["accuracy"])
    # prt_msg += "| average message length"
    # prt_msg += " {:.4f}".format(avg_loss_dict["average_len"])
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
            prt_msg += translations[i].encode('utf-8') + "\n"
        prt_msg += "\n"
    return prt_msg