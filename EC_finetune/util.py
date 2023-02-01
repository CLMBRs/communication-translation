import random
import json
import numpy as np
from typing import Union

import torch
from torch import Tensor
from transformers import PreTrainedTokenizer


TEXT="text"
IMAGE="image"

def get_coco_idx():
    a, b = 56644, 56643
    a_, b_ = [], []

    cand = 0
    while len(a_) < 14500:
        if not cand in a_:
            a_.append(cand)
            cand = cand + 4
        else:
            cand += 1
        cand = cand % 56644

    cand = 0
    while len(b_) < 14500:
        if not cand in b_:
            b_.append(cand)
            cand += 4
        else:
            cand += 1
        cand = cand % 56643

    assert (len(set(a_)) == 14500)
    assert (len(set(b_)) == 14500)

    return a_, b_


def check_dataset_sanity(args):
    assert args.dataset == "coco" or args.dataset == "multi30k"
    if args.dataset == "coco":
        assert (args.src, args.trg) == ("en",
                                        "jp") or (args.src,
                                                  args.trg) == ("jp", "en")
    elif args.dataset == "multi30k":
        assert (args.src, args.trg) == ("en",
                                        "de") or (args.src,
                                                  args.trg) == ("de", "en")


def remove_duplicate(data):
    hash_0 = list(np.round(data[:, 0].numpy(), 3))
    hash_1 = list(np.round(data[:, 1].numpy(), 3))
    hash_2 = list(np.round(data[:, 2].numpy(), 3))
    hash_1000 = list(np.round(data[:, 1000].numpy(), 3))
    hash_2046 = list(np.round(data[:, 2046].numpy(), 3))
    hash_2047 = list(np.round(data[:, 2047].numpy(), 3))

    seen_e2i = {}
    string_ = []
    for idx in range(len(hash_0)):
        keystr = str(hash_0[idx]) + '/' + str(hash_1[idx]) + '/' + str(
            hash_2[idx]
        ) + '/' + str(hash_1000[idx]) + '/' + str(hash_2046[idx]
                                                 ) + '/' + str(hash_2047[idx])
        if keystr in seen_e2i:
            string_.append([seen_e2i[keystr], idx])
        else:
            seen_e2i[keystr] = idx

    string_2 = []
    for pair in string_:
        if torch.sum(torch.abs(data[pair[0]] - data[pair[1]])).numpy() < 15:
            string_2.append(pair)
    s = set([i[-1] for i in string_2])
    index_ = []
    for i in range(len(data)):
        if i not in s:
            index_.append(i)
    data = torch.index_select(data, 0, torch.tensor(index_, dtype=torch.int64))
    return data[:-10000], data[-10000:]


def vocab_constraint_from_file(
    tokenizer: PreTrainedTokenizer,
    file: str,
    threshold: float = 0.99,
    mode="tensor"
) -> Union[Tensor, list]:
    """
    Import a datafile of token frequencies to create a mask for constrained
    generation

    Args:
        file: the datafile to be imported (in json format)
        threshold: the proportion of the total token mass under which tokens
            should be masked
    Returns:
        return a list of bad words' ids (if mode == "list")
        return a tensor where bad words' location has -inf and 0 otherwise (if mode == 'tesnor')
    """
    token_counts = list(json.load(open(file, "r")).items())
    total_count = sum(freq for _, freq in token_counts)
    max_freq = threshold * total_count
    token_counts.sort(key=lambda x: x[1], reverse=True)

    cumulative_freq = 0
    last_index = 0
    for idx, freq in enumerate([x[1] for x in token_counts]):
        cumulative_freq += freq
        if cumulative_freq > max_freq:
            last_index = idx
            break

    # Keep tokens that are in the top probability mass
    good_token_ids = set(
        int(token_id) for token_id, _ in token_counts[:last_index]
    )
    # Leo's comment: this is added for generation purpose; EOS and BOS are valid special tokens to be generated.
    good_token_ids.update(set([tokenizer.eos_token_id, tokenizer.bos_token_id]))

    bad_token_ids = []
    for k, v in tokenizer.get_vocab().items():
        if v in good_token_ids:
            continue
        bad_token_ids.append(v)
    if mode == "list":
        return [[v] for v in bad_token_ids]
    elif mode == "tensor":
        mask = torch.zeros(len(tokenizer))
        mask[bad_token_ids] = -float("inf")
        return mask
    else:
        raise NotImplementedError
