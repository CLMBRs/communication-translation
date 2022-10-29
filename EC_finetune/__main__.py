import argparse
from argparse import Namespace
from bdb import set_trace
import json
import logging
import os
import csv
import sys
from collections import defaultdict
from copy import deepcopy
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import transformers
import yaml
import hydra
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartTokenizer
from omegaconf import DictConfig, OmegaConf, open_dict

from .agents import ImageCaptionGrounder, ECImageIdentificationAgent
from .modelings.modeling_mbart import (
    MBartForCausalLanguageModeling, MBartForConditionalGeneration
)
from .senders import MBartSender, RnnSender
from .receivers import MBartReceiver, RnnReceiver
from .dataloader import (
    CaptionTrainingDataset, TextInputECDataset, XLImageIdentificationDataset
)
from Util.util import create_logger, set_seed, statbar_string
from .util import TEXT, IMAGE

EC_CSV_HEADERS = [
    'mode', 'epoch', 'global step', 'loss', 'accuracy', 'mean_length'
]
CAPTIONING_CSV_HEADERS = [
    'mode', 'epoch', 'global step', 'loss', 'caption generation loss',
    'image selection loss', 'accuracy'
]


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


def evaluate(args, model, dataloader, device, epoch=0, global_step=0):
    batchwise_stats = defaultdict(list)
    max_eval_batches = None
    if hasattr(args, "max_eval_batches"):
        max_eval_batches = args.train_eval.max_eval_batches
    epoch_iterator = tqdm(dataloader, desc='iteration')
    output_ids = []
    for i, batch in enumerate(epoch_iterator):
        if max_eval_batches and i >= max_eval_batches:
            break
        model.eval()
        batch['sender_image'] = batch['sender_image'].to(device)
        batch['receiver_images'] = batch['receiver_images'].to(device)
        batch['target'] = batch['target'].to(device)
        if args.mode == 'image_grounding':
            batch['caption_ids'] = batch['caption_ids'].to(device)
            batch['caption_mask'] = batch['caption_mask'].to(device)

        eval_return_dict = model(batch)

        if 'message' in eval_return_dict:
            output_ids.append(eval_return_dict['message'].cpu().detach().numpy())

        eval_return_dict['loss'] = eval_return_dict['loss'].item()
        for key, value in eval_return_dict.items():
            if key in args.train_eval.stats_to_print:
                batchwise_stats[key].append(value)

    average_stats = {}
    average_stats['epoch'] = epoch
    average_stats['global step'] = global_step
    average_stats['mode'] = 'validation'
    for key, value in batchwise_stats.items():
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
        if not (k.startswith("language_model") or k.startswith("orig_model"))
    }
    torch.save(state_dict, args.output_dir + "/model.pt")

    # For pretrained models, provide extra saving strategy
    if args.save_pretrain_seperately:
        # Save a trained model, configuration and tokenizer
        # using `save_pretrained()`. They can then be
        # reloaded using `from_pretrained()`
        model_to_save = (
            model.sender.sender.module
            if hasattr(model.sender.top, 'module') else model.sender.top
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)


def train(args, model, dataloader, valid_dataloader, tokenizer, params, logger, device):
    global_step = 0
    best_loss = np.inf
    checkpoint_stats = defaultdict(list)
    gradient_count = 0
    train_csv_data = []
    val_csv_data = []

    optimizer = torch.optim.Adam(params, lr=args.train_eval.lr)
    if args.train_eval.schedule == 'linear_w_warmup':
        scheduler_method = transformers.get_linear_schedule_with_warmup
        scheduler_args = {
            'optimizer': optimizer,
            'num_warmup_steps': args.train_eval.num_warmup_steps,
            'num_training_steps': args.train_eval.max_global_step
        }
    else:
        # Default to constant schedule with warmup
        scheduler_method = transformers.get_constant_schedule_with_warmup
        scheduler_args = {
            'optimizer': optimizer,
            'num_warmup_steps': args.train_eval.num_warmup_steps
        }
    scheduler = scheduler_method(**scheduler_args)

    for epoch in range(args.train_eval.num_games):
        epoch_iterator = tqdm(dataloader, desc='Iteration')
        for batch in epoch_iterator:
            model.train()
            # Move data to the GPU
            for key, value in batch.items():
                if isinstance(value, Tensor):
                    batch[key] = value.to(device)

            train_return_dict = model(batch)
            loss = train_return_dict['loss']
            optimizer.zero_grad()
            loss.backward()
            gradient_count += 1
            nn.utils.clip_grad_norm_(params, args.train_eval.grad_clip)
            if gradient_count >= args.train_eval.gradient_accumulation_steps:
                optimizer.step()
                scheduler.step()
                gradient_count = 0
                global_step += 1

            train_return_dict['loss'] = train_return_dict['loss'].item()
            for key, value in train_return_dict.items():
                if key in args.train_eval.stats_to_print:
                    checkpoint_stats[key].append(value)

            if global_step % args.train_eval.print_every == 0 and gradient_count == 0:
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

            if global_step % args.train_eval.valid_every == 0 and gradient_count == 0:
                with torch.no_grad():
                    results, output_ids, printout = evaluate(
                        args, model, valid_dataloader, device, epoch, global_step
                    )
                    # import pbd; pdb.set_trace()
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
                        if args.data.save_output_txt:
                            output_texts = ids_to_texts(output_ids, tokenizer)
                            with open(
                                args.output_dir + "/eval_texts.txt", 'w'
                            ) as f:
                                for i in output_texts:
                                    f.write(i)
                        save(args, model, logger)
                        print(
                            f"Epoch: {epoch}, Prediction Accuracy: {cur_acc},"
                            f" Saved to Path: {args.output_dir}"
                        )
                        if cur_acc > args.train_eval.target_acc and args.generation.TransferH:
                            args.generation.hard = True

            if global_step >= args.train_eval.max_global_step:
                return global_step

def namespaced_hparams(hparams):
    if type(hparams) is not dict:
        return hparams
    else:
        hparams = {k: namespaced_hparams(v)
            for k, v in hparams.items()}
        return Namespace(**hparams)

@hydra.main(version_base=None, config_path="../Configs")
def main(args: DictConfig):
    """
    Train a model to generate image captions
    """
    
    # set csv output file
    # import pdb; pdb.set_trace()
    container = OmegaConf.to_object(args)
    # turn hparams into a namespace
    args = namespaced_hparams(container['ec'])
    if not os.path.exists(args.output_dir):
        # TODO: move output_dir to ec?
        os.makedirs(args.output_dir)

    # Add new field
    if args.mode == 'image_grounding':
        args.csv_headers = CAPTIONING_CSV_HEADERS
    else:
        args.csv_headers = EC_CSV_HEADERS
        args.csv_headers += ['communication loss']
        if args.train_eval.language_model_lambda:
            args.csv_headers += ['lm loss']
        if args.train_eval.weight_drift_lambda:
            args.csv_headers += ['drift loss']

    with open(f"{args.output_dir}/log.csv", 'w') as f:
        csv_file = csv.DictWriter(f, fieldnames=args.csv_headers)
        csv_file.writeheader()

    logging.info("Entering main run script")

    # Setup CUDA, GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(args.train_eval.seed)
    # Set logger name properly
    logger_name = ""
    logger_name += "caption" if args.mode == 'image_grounding' else "ec_finetune"
    # from pdb import set_trace; set_trace()
    logger_name += "(T2I)" if args.train_eval.sender_input_type == TEXT else "(I2I)"
    logger = create_logger(name=logger_name)
    
    # Load dataset
    if args.mode == 'image_grounding' or args.train_eval.sender_input_type == TEXT:
        train_captions = [
            [caption.strip() for caption in json.loads(line)]
            for line in open(args.data.train_captions, 'r').readlines()
        ] # [:num_train]
        valid_captions = [
            [caption.strip() for caption in json.loads(line)]
            for line in open(args.data.valid_captions, 'r').readlines()
        ]
    train_images = torch.load(args.data.train_images)
    valid_images = torch.load(args.data.valid_images)

    logger.info("Dataset Loaded")

    # Write the model description
    logger.info("Configuration:")
    print(args)

    tokenizer = MBartTokenizer.from_pretrained(args.model.model_name)
    args.model.tokenizer = tokenizer
    tokenizer.save_pretrained(args.output_dir)
    vocab = tokenizer.get_vocab()
    args.model.padding_index = vocab['<pad>']
    args.model.cls_index = vocab['<s>']
    args.model.vocab_size = len(vocab)

    language_model = None
    orig_model = None

    # Initialize Sender and Receiver, either from pretrained Bart or as a
    # from-scratch RNN
    if args.model.model_name == 'rnn':
        comm_model = nn.GRU(
            input_size=args.model.hidden_dim,
            hidden_size=args.model.hidden_dim,
            num_layers=args.model.num_layers,
            batch_first=True
        )
        sender = RnnSender(
            comm_model, args.model.hidden_dim, args.model.vocab_size, args.model.bos_idx
        )
        receiver = RnnReceiver(comm_model, args.model.hidden_dim, args.model.vocab_size)
    else:
        # If language modeling loss is to be used, get a copy of the original
        # facebook weights, deepcopy the decoder and embeddings, and delete the
        # rest
        if args.train_eval.language_model_lambda:
            language_model = MBartForCausalLanguageModeling.from_pretrained(
                args.train_eval.language_model_path
            )
            for param in language_model.parameters():
                param.requires_grad = False

        comm_model = MBartForConditionalGeneration.from_pretrained(
            args.model.model_name
        )

        if args.train_eval.weight_drift_lambda:
            orig_model = deepcopy(comm_model)
            for param in orig_model.parameters():
                param.requireds_grad = False

        sender = MBartSender(
            comm_model,
            seq_len=args.generation.max_seq_length,
            unroll=args.model.image_unroll,
            unroll_length=args.model.image_unroll_length,
            temperature=args.generation.temperature,
            hard=args.generation.hard,
            repetition_penalty=args.generation.repetition_penalty,
            beam_width=args.generation.beam_width,
            generate_from_logits=args.generation.generate_from_logits,
            sender_input_type=args.train_eval.sender_input_type
        )
        receiver = MBartReceiver(
            comm_model,
            recurrent_aggregation=args.model.recurrent_hidden_aggregation,
            dropout=args.model.dropout,
            unit_norm=args.model.unit_norm
        )

    # Initialize agent setup
    if args.mode == 'image_grounding':
        model = ImageCaptionGrounder(sender, receiver, args)
        training_set = CaptionTrainingDataset(
            train_images,
            train_captions,
            args.train_eval.num_distractors_train,
            tokenizer,
            args,
            max_length=args.generation.max_seq_length
        )
        valid_set = CaptionTrainingDataset(
            valid_images,
            valid_captions,
            args.train_eval.num_distractors_valid,
            tokenizer,
            args,
            max_length=args.generation.max_seq_length
        )
    else:
        model = ECImageIdentificationAgent(
            sender,
            receiver,
            args,
            language_model=language_model,
            orig_model=orig_model
        )
        if args.train_eval.sender_input_type == TEXT:
            print("*T2I* EC training")
            training_set = TextInputECDataset(
                train_images,
                train_captions,
                args.num_distractors_train,
                tokenizer,
                args,
                max_length=args.max_text_seq_length
            )
            valid_set = TextInputECDataset(
                valid_images,
                valid_captions,
                args.num_distractors_valid,
                tokenizer,
                args,
                max_length=args.max_text_seq_length,
                max_captions_per_image=1
            )
        else:
            print("*I2I* EC training")
            training_set = XLImageIdentificationDataset(
                train_images, args.train_eval.num_distractors_train, args, tokenizer
            )
            valid_set = XLImageIdentificationDataset(
                valid_images, args.train_eval.num_distractors_valid, args, tokenizer
            )

    if args.model.load_entire_agent:
        state_dict = torch.load(args.model.model_name + "/model.pt")
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

    if args.model.freeze_adapters:
        print("Freezing adapter modules")
        model.freeze_adapters()

    if args.model.freeze_sender:
        print("Freezing sender's decoder")
        model.freeze_sender_decoder()

    if args.model.freeze_receiver:
        print("Freezing listener's encoder")
        model.freeze_listener_encoder()

    # Move the model to gpu if the configuration calls for it
    model.to(device)

    # Initialize the dataloader
    training_params = {
        'batch_size': args.train_eval.batch_size,
        'shuffle': True,
        'drop_last': True
    }
    test_params = {
        'batch_size': args.train_eval.batch_size,
        'shuffle': False,
        'drop_last': False
    }

    training_dataloader = DataLoader(training_set, **training_params)
    valid_dataloader = DataLoader(valid_set, **test_params)

    if args.train_eval.do_train:
        train(
            args, model, training_dataloader, valid_dataloader, tokenizer,
            model.parameters(), logger, device
        )
    if args.train_eval.do_eval:
        checkpoint = args.output_dir + "/model.pt"
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model.load_state_dict(torch.load(checkpoint), strict=False)
        model.to(device)
        model.eval()
        results, output_ids, printout = evaluate(args, model, valid_dataloader, device)
        if args.data.save_output_txt:
            output_texts = ids_to_texts(output_ids, tokenizer)
            with open(args.output_dir + "/eval_texts.txt", 'w') as f:
                for i in output_texts:
                    f.write(i)

        logger.info("Best model stats: ")
        logger.info(printout)


if __name__ == '__main__':
    main()
