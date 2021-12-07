import argparse
import json
import logging
import os
import csv
import random
import sys
from collections import defaultdict
from copy import deepcopy
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import transformers
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartTokenizer

from EC_finetune.agents import ImageCaptionGrounder, ECImageIdentificationAgent
from EC_finetune.modelings.modeling_mbart import MBartForConditionalGeneration
from EC_finetune.senders import MBartSender, RnnSender
from EC_finetune.receivers import MBartReceiver, RnnReceiver
from EC_finetune.dataloader import (
    CaptionTrainingDataset, XLImageIdentificationDataset
)

EC_CSV_HEADERS = [
    "mode", "epoch", "global step", "loss", "accuracy", "mean_length"
]
EC_LM_CSV_HEADERS = [
    "mode", "epoch", "global step", "loss", "lm loss", "communication loss", "accuracy", "mean_length"
]
CAPTIONING_CSV_HEADERS = [
    "mode", "epoch", "global step", "loss", "caption generation loss",
    "image selection loss", "accuracy"
]


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def ids_to_texts(output_ids, tokenizer):
    text = []
    # concatnate batch-wise ids
    # Sometimes give ValueError: could not broadcast input array from shape
    # (8,64) into shape (8,)
    # output_ids = np.concatenate(np.array(output_ids), axis=0)
    for batch in output_ids:
        for i in batch:
            text.append(tokenizer.decode(i) + '\n')
    return text


def statbar_string(stat_dict: dict) -> str:
    """
    Return a printable "statbar" string from a dictionary of named statistics
    """
    stat_items = []
    for key, value in stat_dict.items():
        stat_items.append(f"{key} {value}")
    return ' | '.join(stat_items)


def evaluate(args, model, dataloader, epoch=0, global_step=0):
    stats = defaultdict(list)
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    output_ids = []
    for batch in epoch_iterator:
        # Start evaluation mode
        model.eval()

        # Move data to the GPU
        batch['sender_image'] = batch['sender_image'].to(args.device)
        batch['receiver_images'] = batch['receiver_images'].to(args.device)
        batch['target'] = batch['target'].to(args.device)
        if args.mode == 'image_grounding':
            batch['caption_ids'] = batch['caption_ids'].to(args.device)
            batch['caption_mask'] = batch['caption_mask'].to(args.device)

        eval_return_dict = model(batch)

        output_ids.append(eval_return_dict['message'].cpu().detach().numpy())

        eval_return_dict['loss'] = eval_return_dict['loss'].item()
        for key, value in eval_return_dict.items():
            if key in args.stats_to_print:
                stats[key].append(value)

    average_stats = {}
    average_stats['epoch'] = epoch
    average_stats['global step'] = global_step
    average_stats['mode'] = 'validation'
    for key, value in stats.items():
        average_stats[key] = round(mean(value), 4)

    printout = statbar_string(average_stats)

    return average_stats, output_ids, printout


def save(args, model, logger):
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Saving model to %s", args.output_dir)

    # Good practice: save your training arguments together
    # with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    # Save the general part of the model
    state_dict = {
        k: v
        for k, v in model.state_dict().items()
        if not k.startswith("language_model")
    }
    torch.save(state_dict, args.output_dir + '/model.pt')

    # For pretrained models, provide extra saving strategy
    if args.save_pretrain_seperately:
        # Save a trained model, configuration and tokenizer
        # using `save_pretrained()`. They can then be
        # reloaded using `from_pretrained()`
        model_to_save = (
            model.sender.sender.module
            if hasattr(model.sender.top, "module") else model.sender.top
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)


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

    for epoch in range(args.num_games):
        epoch_iterator = tqdm(dataloader, desc="Iteration")
        for batch in epoch_iterator:
            model.train()
            # Move data to the GPU
            batch['sender_image'] = batch['sender_image'].to(args.device)
            batch['receiver_images'] = batch['receiver_images'].to(args.device)
            batch['target'] = batch['target'].to(args.device)
            if args.mode == 'image_grounding':
                batch['caption_ids'] = batch['caption_ids'].to(args.device)
                batch['caption_mask'] = batch['caption_mask'].to(args.device)

            train_return_dict = model(batch)
            loss = train_return_dict['loss']
            optimizer.zero_grad()
            loss.backward()
            gradient_count += 1
            nn.utils.clip_grad_norm_(params, args.grad_clip)
            if gradient_count >= args.gradient_accumulation_steps:
                optimizer.step()
                scheduler.step()
                gradient_count = 0
                global_step += 1

            train_return_dict['loss'] = train_return_dict['loss'].item()
            for key, value in train_return_dict.items():
                if key in args.stats_to_print:
                    checkpoint_stats[key].append(value)

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
                    results, output_ids, printout = evaluate(
                        args, model, valid_dataloader, epoch, global_step
                    )
                    val_csv_data.append(results)
                    with open(f"{args.output_dir}/log.csv", 'a') as f:
                        csv_file = csv.DictWriter(
                            f, fieldnames=args.csv_headers
                        )
                        csv_file.writerow(results)
                    # Output evaluation statistics
                    logger.info(printout)
                    cur_acc = float(results['accuracy'])
                    cur_loss = float(results['loss'])
                    if cur_loss < best_loss:
                        best_loss = cur_loss
                        save(args, model, logger)
                        print(
                            f"Epoch: {epoch}, Prediction Accuracy: {cur_acc},"
                            f" Saved to Path: {args.output_dir}"
                        )
                        if cur_acc > args.target_acc and args.TransferH:
                            args.hard = True

            if global_step >= args.max_global_step:
                return global_step

    print(f"Length of train data logs: {len(train_csv_data)}")
    print(f"Length of val data logs: {len(val_csv_data)}")


def main():
    """
    Train a model to generate image captions
    """

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
    parser = argparse.ArgumentParser(description='Image caption training')
    parser.add_argument('--config', type=str)
    parser.add_argument('--seed_override', type=int)
    args = parser.parse_args()
    args_dict = vars(args)
    with open(args_dict['config'], 'r') as config_file:
        args_dict.update(yaml.load(config_file, Loader=yaml.FullLoader))

    # set random seed
    if args.seed_override:
        args.seed = args.seed_override
    set_seed(args)

    # set csv output file
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.mode == 'image_grounding':
        args.csv_headers = CAPTIONING_CSV_HEADERS
    elif args.language_model_loss:
        args.csv_headers = EC_LM_CSV_HEADERS
    else:
        args.csv_headers = EC_CSV_HEADERS
        if args.language_model_loss or args.weight_drift_loss:
            args.csv_headers += ["communication loss"]
        if args.language_model_loss:
            args.csv_headers += ["lm loss"]
        if args.weight_drift_loss:
            args.csv_headers += ["drift loss"]

    with open(f"{args.output_dir}/log.csv", 'w') as f:
        csv_file = csv.DictWriter(f, fieldnames=args.csv_headers)
        csv_file.writeheader()

    logging.info('Entering main run script')

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    if args.mode == 'image_grounding':
        train_captions = [
            json.loads(line)
            for line in open(args.train_captions, 'r').readlines()
        ]
        valid_captions = [
            json.loads(line)
            for line in open(args.valid_captions, 'r').readlines()
        ]
    train_images = torch.load(args.train_images)
    valid_images = torch.load(args.valid_images)

    logger.info('Dataset Loaded')

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

    language_model = None
    orig_model = None

    # Initialize Sender and Receiver, either from pretrained Bart or as a
    # from-scratch RNN
    if args.model_name == 'rnn':
        comm_model = nn.GRU(
            input_size=args.hidden_dim,
            hidden_size=args.hidden_dim,
            num_layers=args.num_layers,
            batch_first=True
        )
        sender = RnnSender(
            comm_model, args.hidden_dim, args.vocab_size, args.bos_idx
        )
        receiver = RnnReceiver(comm_model, args.hidden_dim, args.vocab_size)
    else:
        # If language modeling loss is to be used, get a copy of the original
        # facebook weights, deepcopy the decoder and embeddings, and delete the
        # rest
        if args.language_model_loss:
            tmp_language_model = MBartForConditionalGeneration.from_pretrained(
                "facebook/mbart-large-cc25"
            )

            decoder = deepcopy(tmp_language_model.model.decoder)
            embedding = deepcopy(tmp_language_model.model.shared)
            bias = deepcopy(tmp_language_model.final_logits_bias)

            del tmp_language_model

            for param in decoder.parameters():
                param.requires_grad = False
            for param in embedding.parameters():
                param.requires_grad = False
            bias.requires_grad = False

            language_model = {
                'decoder': decoder.to(device),
                'embedding': embedding.to(device),
                'bias': bias.to(device)
            }

        comm_model = MBartForConditionalGeneration.from_pretrained(
            args.model_name
        )

        if args.weight_drift_loss:
            orig_model = deepcopy(comm_model)
            for param in orig_model.parameters():
                param.requireds_grad = False

        sender = MBartSender(
            comm_model,
            args.hidden_dim,
            seq_len=args.max_seq_length,
            recurrent_unroll=args.recurrent_image_unroll,
            temperature=args.temperature,
            hard=args.hard,
            repetition_penalty=args.repetition_penalty,
            beam_width=args.beam_width,
            generate_from_logits=args.generate_from_logits
        )
        receiver = MBartReceiver(
            comm_model,
            args.hidden_dim,
            recurrent_aggregation=args.recurrent_hidden_aggregation,
            dropout=args.dropout,
            unit_norm=args.unit_norm
        )

    # Initialize agent setup
    if args.mode == 'image_grounding':
        model = ImageCaptionGrounder(sender, receiver, args)
        training_set = CaptionTrainingDataset(
            train_images,
            train_captions,
            args.num_distractors_train,
            tokenizer,
            args,
            max_length=args.max_seq_length
        )
        valid_set = CaptionTrainingDataset(
            valid_images,
            valid_captions,
            args.num_distractors_valid,
            tokenizer,
            args,
            max_length=args.max_seq_length
        )
    else:
        model = ECImageIdentificationAgent(
            sender, receiver, args, language_model=language_model
        )
        training_set = XLImageIdentificationDataset(
            train_images, args.num_distractors_train, args, tokenizer
        )
        valid_set = XLImageIdentificationDataset(
            valid_images, args.num_distractors_valid, args, tokenizer
        )

    if args.load_entire_agent:
        state_dict = torch.load(args.model_name + "/model.pt")
        state_dict = {
            k: v
            for k, v in state_dict.items() if (
                not (
                    k.startswith("sender.top") or
                    k.startswith("sender.decoder") or
                    k.startswith("sender.embedding") or
                    k.startswith("sender.output_bias") or
                    k.startswith("receiver.encoder") or
                    k.startswith("receiver.embedding")
                )
            )
        }
        model.load_state_dict(state_dict, strict=False)

    # Move the model to gpu if the configuration calls for it
    model.to(args.device)

    # Initialize the dataloader
    training_params = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "drop_last": True
    }
    test_params = {
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False
    }

    training_dataloader = DataLoader(training_set, **training_params)
    valid_dataloader = DataLoader(valid_set, **test_params)

    if args.do_train:
        train(
            args, model, training_dataloader, valid_dataloader,
            model.parameters(), logger
        )
    if args.do_eval:
        checkpoint = args.output_dir + '/model.pt'
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model.load_state_dict(torch.load(checkpoint), strict=False)
        model.to(args.device)
        model.eval()
        results, output_ids, printout = evaluate(args, model, valid_dataloader)
        if args.save_output_txt:
            output_texts = ids_to_texts(output_ids, tokenizer)
            with open(args.output_dir + '/eval_texts.txt', 'w') as f:
                for i in output_texts:
                    f.write(i)

        logger.info("Best model stats: ")
        logger.info(printout)


if __name__ == '__main__':
    main()
