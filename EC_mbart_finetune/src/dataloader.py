import copy
import json
import operator
import pickle as pkl
import numpy as np
from collections import OrderedDict
import time
import torch
from torch.autograd import Variable
# from torchfile import load as load_lua

from EC_mbart_finetune.src.util import *

random = np.random
random.seed(1234)


def next_batch_joint(images, batch_size, num_distractor, lang_info, tt):
    """
    This function create batched images for listener and speaker, including creation of distractors

    :param images: all the image to select from
    :param batch_size:
    :param num_distractor: number of distractors in each game (not considering batch)
    :param tt: torch backend, whether it is cpu or cuda
    :param lang_info: [(lang_id1, lang_id2), (lang_mask1, lang_mask2)]
    :return:
    """
    # import pdb; pdb.set_trace()
    spk_imgs, spk_caps, lsn_imgs, lsn_caps, correct_indices = [], [], [], [], []
    spk_lang_ids, spk_lang_masks = [], []
    lang_masks = None
    if len(lang_info) == 2:
        lang_ids, lang_masks = lang_info
    else:
        lang_ids, = lang_info
    assert len(lang_ids) == len(lang_masks) == 2
    total_indices = []
    keys = range(len(images))
    assert len(keys) >= num_distractor
    for batch_idx in range(batch_size):

        img_indices = random.permutation(len(images))[:num_distractor]

        correct_idx = random.randint(0, num_distractor)  # (1)

        spk_img = img_indices[correct_idx]
        spk_imgs.append(spk_img)  # (batch_size, 2048)

        # choose a language to generate sentence for
        random_lang_idx = np.random.choice([0, 1])
        chosen_lang_id = lang_ids[random_lang_idx]
        spk_lang_ids.append(chosen_lang_id)
        if lang_masks is not None:
            chosen_lang_mask = lang_masks[random_lang_idx]
            spk_lang_masks.append(chosen_lang_mask)

        lsn_imgs += list(img_indices)  # batch_size * num_dist

        correct_indices.append(correct_idx)  # (batch_size)

    spk_imgs = torch.index_select(images, 0, torch.tensor(spk_imgs))
    spk_imgs = torch.tensor(spk_imgs, requires_grad=False).view(batch_size, -1)
    lsn_imgs = torch.index_select(images, 0, torch.tensor(lsn_imgs)).view(
        batch_size, num_distractor, -1
    )
    lsn_imgs = torch.tensor(lsn_imgs, requires_grad=False)
    spk_lang_ids = torch.tensor(spk_lang_ids, requires_grad=False)
    spk_lang_masks = torch.stack(spk_lang_masks
                                ) if len(spk_lang_masks) > 0 else torch.tensor(
                                    []
                                )

    correct_indices = torch.tensor(correct_indices,
                                   requires_grad=False).view(batch_size)
    assert correct_indices.dtype == torch.int64  # LongTensor
    if torch.cuda.is_available():
        spk_imgs = spk_imgs.cuda()
        lsn_imgs = lsn_imgs.cuda()
        spk_lang_ids = spk_lang_ids.cuda()
        spk_lang_masks = spk_lang_masks.cuda()
        correct_indices = correct_indices.cuda()
    return (
        spk_imgs, lsn_imgs, 0, 0, spk_lang_ids, spk_lang_masks, 0, 0, 0,
        correct_indices
    )


def weave_out(caps_out):
    ans = []
    seq_len = max([len(x) for x in caps_out])
    for idx in range(seq_len):
        for sublst in caps_out:
            if idx < len(sublst):
                ans.append(sublst[idx])
    return ans
