import argparse
import csv
import logging
import os
import sys
from collections import defaultdict
from statistics import mean

import transformers
import yaml
import numpy as np
from tqdm import tqdm
from transformers import MBartTokenizer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from EC_finetune.dataloader import XLMDataset, SingleLangXLMDataset
from EC_finetune.modelings.modeling_mbart import MBartForCausalLanguageModeling
from EC_finetune.util import set_seed, statbar_string


def evaluate(args, model, dataloader, epoch=0, global_step=0):
    batchwise_stats = defaultdict(list)
    epoch_iterator = tqdm(dataloader, desc='iteration')
    for batch in epoch_iterator:
        model.eval()
        batch['input_ids'] = batch['input_ids'].squeeze().to(args.device)
        batch['attention_mask'] = batch['attention_mask'].squeeze().to(
            args.device
        )
        targets = batch['input_ids']
        outputs = model(**batch)
        loss = F.cross_entropy(
            outputs.transpose(1, 2), targets, ignore_index=args.padding_index
        )
        loss = loss.item()
        batchwise_stats['loss'].append(loss)

    average_stats = {}
    average_stats['epoch'] = epoch
    average_stats['global step'] = global_step
    average_stats['mode'] = 'validation'
    for key, value in batchwise_stats.items():
        average_stats[key] = round(mean(value), 4)
    printout = statbar_string(average_stats)

    return average_stats, printout


def train(args, model, dataloader, valid_dataloader, params, logger):
    global_step = 0
    best_loss = np.inf
    checkpoint_stats = defaultdict(list)
    gradient_count = 0
    train_csv_data = []
    val_csv_data = []

    optimizer = torch.optim.Adam(params, lr=args.lr)
    if args.schedule == 'linear_w_warmup':
        scheduler_method = transformers.get_linear_schedule_with_warmup
        scheduler_args = {
            'optimizer': optimizer,
            'num_warmup_steps': args.num_warmup_steps,
            'num_training_steps': args.max_global_step
        }
    else:
        # Default to constant schedule with warmup
        scheduler_method = transformers.get_constant_schedule_with_warmup
        scheduler_args = {
            'optimizer': optimizer,
            'num_warmup_steps': args.num_warmup_steps
        }
    scheduler = scheduler_method(**scheduler_args)

    for epoch in range(args.num_epochs):
        epoch_iterator = tqdm(dataloader, desc="iteration")
        for batch in epoch_iterator:
            model.train()
            # Move data to the GPU
            batch['input_ids'] = batch['input_ids'].squeeze().to(args.device)
            batch['attention_mask'] = batch['attention_mask'].squeeze().to(
                args.device
            )
            targets = batch['input_ids']
            outputs = model(**batch)
            loss = F.cross_entropy(
                outputs.transpose(1, 2),
                targets,
                ignore_index=args.padding_index
            )
            optimizer.zero_grad()
            loss.backward()
            gradient_count += 1
            nn.utils.clip_grad_norm_(params, args.grad_clip)
            if gradient_count >= args.gradient_accumulation_steps:
                optimizer.step()
                scheduler.step()
                gradient_count = 0
                global_step += 1

            loss = loss.item()
            checkpoint_stats['loss'].append(loss)

            if global_step % args.print_every == 0 and gradient_count == 0:
                checkpoint_average_stats = {}
                checkpoint_average_stats['epoch'] = epoch
                checkpoint_average_stats['global step'] = global_step
                checkpoint_average_stats['mode'] = 'train'
                for key, value in checkpoint_stats.items():
                    checkpoint_average_stats[key] = round(mean(value), 4)
                with open(f"{args.output_dir}/log.csv", 'a') as f:
                    csv_file = csv.DictWriter(f, fieldnames=args.csv_headers)
                    csv_file.writerow(checkpoint_average_stats)
                train_csv_data.append(checkpoint_stats)

                logger.info(statbar_string(checkpoint_average_stats))
                checkpoint_stats = defaultdict(list)

            if global_step % args.valid_every == 0 and gradient_count == 0:
                with torch.no_grad():
                    eval_return_dict, printout = evaluate(
                        args, model, valid_dataloader, epoch, global_step
                    )
                    val_csv_data.append(eval_return_dict)
                    with open(f"{args.output_dir}/log.csv", 'a') as f:
                        csv_file = csv.DictWriter(
                            f, fieldnames=args.csv_headers
                        )
                        csv_file.writerow(eval_return_dict)
                    logger.info(printout)
                    cur_loss = float(eval_return_dict['loss'])
                    if cur_loss < best_loss:
                        best_loss = cur_loss
                        if not os.path.exists(args.output_dir):
                            os.makedirs(args.output_dir)
                        model.save_pretrained(args.output_dir)

            if global_step >= args.max_global_step:
                return


def main():
    # Configure the logger (boilerplate)
    logger = logging.getLogger(__name__)
    out_handler = logging.StreamHandler(sys.stdout)
    message_format = '%(asctime)s - %(message)s'
    date_format = '%m-%d-%y %H:%M:%S'
    out_handler.setFormatter(logging.Formatter(message_format, date_format))
    out_handler.setLevel(logging.INFO)
    logger.addHandler(out_handler)
    logger.setLevel(logging.INFO)

    # Parse command line arguments (essentially only the configuration file,
    # which is read into a dictionary)
    parser = argparse.ArgumentParser(
        description="Script to finetune an mbart causal LM"
    )
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    args_dict = vars(args)
    with open(args_dict['config'], 'r') as config_file:
        args_dict.update(yaml.load(config_file, Loader=yaml.FullLoader))

    set_seed(args.seed, args.n_gpu)

    with open(f"{args.output_dir}/log.csv", 'w') as f:
        csv_file = csv.DictWriter(f, fieldnames=args.csv_headers)
        csv_file.writeheader()

    # Setup CUDA, GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device

    # Write the model description
    logger.info('Configuration:')
    print(args)

    tokenizer = MBartTokenizer.from_pretrained(args.model_name)
    args.tokenizer = tokenizer
    tokenizer.save_pretrained(args.output_dir)
    vocab = tokenizer.get_vocab()
    args.padding_index = vocab['<pad>']
    args.cls_index = vocab['<s>']
    args.vocab_size = len(vocab)

    model = MBartForCausalLanguageModeling.from_pretrained(args.model_name)

    # Move the model to gpu if the configuration calls for it
    model.to(args.device)

    train_lang_datasets = {
        lang_id: SingleLangXLMDataset(
            os.path.join(args.data_dir, data_file),
            args.batch_size,
            order='sort'
        )
        for lang_id, data_file in args.train_data_files.items()
    }
    valid_lang_datasets = {
        lang_id: SingleLangXLMDataset(
            os.path.join(args.data_dir, data_file), args.batch_size
        )
        for lang_id, data_file in args.valid_data_files.items()
    }

    train_dataset = XLMDataset(
        train_lang_datasets,
        tokenizer,
        alpha=args.lang_alpha,
        max_length=args.max_seq_length
    )
    valid_dataset = XLMDataset(valid_lang_datasets, tokenizer, alpha=1.0)
    train_dataloader = DataLoader(train_dataset, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, shuffle=False)

    train(
        args, model, train_dataloader, valid_dataloader, model.parameters(),
        logger
    )


if __name__ == '__main__':
    main()