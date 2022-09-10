import argparse
import os
import random
from math import ceil

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

def write_splits(args, dataset, source_id, target_id):
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )
    source_lines = []
    target_lines = []
    num_batch = ceil(args.max_set_size / args.batch_size)
    for i, batch in enumerate(dataloader):
        if i == num_batch:
            break
        translation_batch = batch["translation"]
        source_lines += [
            line for line in translation_batch[source_id] if line.strip() != ''
        ]
        target_lines += [
            line for line in translation_batch[target_id] if line.strip() != ''
        ]
    source_filename = f"{args.lang_pair}.{source_id}.{args.split}"
    target_filename = f"{args.lang_pair}.{target_id}.{args.split}"
    source_file = os.path.join(args.output_dir, source_filename)
    target_file = os.path.join(args.output_dir, target_filename)
    with open(source_file, 'w+') as f:
        for line in source_lines:
            print(line, file=f)
    with open(target_file, 'w+') as f:
        for line in target_lines:
            print(line, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Write splits from dataset script")

    parser.add_argument('--dataset_script', type=str)
    parser.add_argument('--lang_pair', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--lang1_id', type=str)
    parser.add_argument('--lang2_id', type=str)
    parser.add_argument('--max_set_size', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()
    args_dict = vars(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    dataset = load_dataset(
        args.dataset_script, args.lang_pair, split=args.split
    )
        
    write_splits(args, dataset, args.lang1_id, args.lang2_id)
