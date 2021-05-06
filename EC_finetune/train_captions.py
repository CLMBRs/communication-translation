import argparse
import json
import logging
import os
import random
import sys
from collections import defaultdict
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartTokenizer

from EC_finetune.agents import CommunicationAgent
from EC_finetune.dataloader import ImageCaptionDataset
from EC_finetune.util import print_loss_, remove_duplicate


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def ids_to_texts(output_ids, tokenizer):
    text = []
    # concatnate batch-wise ids
    output_ids = np.concatenate(np.array(output_ids), axis=0)
    for i in output_ids:
        text.append(tokenizer.decode(i) + '\n')
    return text


def evaluate(args, model, dataloader, epoch=0):
    stats = defaultdict(list)
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    output_ids = []
    for batch in epoch_iterator:
        # Start evaluation mode
        model.eval()

        # Move data to the GPU
        batch['speaker_image'] = batch['speaker_image'].to(args.device)
        batch['listener_images'] = batch['listener_images'].to(args.device)
        batch['target'] = batch['target'].to(args.device)

        eval_return_dict = model(batch)

        output_ids.append(eval_return_dict['message'].cpu().detach().numpy())

        eval_return_dict['loss'] = eval_return_dict['loss'].item()
        eval_return_dict['mean_length'] = eval_return_dict['mean_length'].item()
        for key, value in eval_return_dict.items():
            if key in ['loss', 'accuracy', 'mean_length']:
                stats[key].append(value)

    average_stats = {}
    for key, value in stats.items():
        average_stats[key] = mean(value)

    s_new = print_loss_(epoch, args.alpha, average_stats, 'valid')

    return average_stats, output_ids, s_new


def save(args, model, logger):
    # Create output directory if needed
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    logger.info("Saving model to %s", args.output_dir)

    # Save the general part of the model
    torch.save(model.state_dict(), args.output_dir + '/model.pt')
    # Good practice: save your training arguments together
    # with the trained model
    torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    # For pretrained models, provide extra saving strategy
    if args.save_pretrain_seperately:
        # Save a trained model, configuration and tokenizer
        # using `save_pretrained()`. They can then be
        # reloaded using `from_pretrained()`
        model_to_save = (
            model.model.module if hasattr(model, "module") else model.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)


def train(args, model, dataloader, valid_dataloader, params, logger):
    optimizer = torch.optim.Adam(params, lr=args.lr)
    global_step = 0
    best_acc = 0.0
    checkpoint_stats = defaultdict(list)

    for epoch in range(args.num_games):
        epoch_iterator = tqdm(dataloader, desc="Iteration")
        for batch in epoch_iterator:
            # Inform the training started.
            model.train()

            # Move data to the GPU
            batch['speaker_image'] = batch['speaker_image'].to(args.device)
            batch['listener_images'] = batch['listener_images'].to(args.device)
            batch['target'] = batch['target'].to(args.device)

            train_return_dict = model(batch)
            loss = train_return_dict['loss']
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, args.grad_clip)
            optimizer.step()
            global_step += 1
            if global_step >= args.max_global_step:
                # Save model even if it does not reach the target acc in the
                # end
                if best_acc == 0:
                    save(args, model, logger)
                return global_step

            train_return_dict['loss'] = loss.item()
            train_return_dict['mean_length'] = (
                train_return_dict['mean_length'].item()
            )

            for key, value in train_return_dict.items():
                if key in ['loss', 'accuracy', 'mean_length']:
                    checkpoint_stats[key].append(value)

            if global_step % args.print_every == 0:
                checkpoint_average_stats = {}
                for key, value in checkpoint_stats.items():
                    checkpoint_average_stats[key] = mean(value)
                logger.info(
                    print_loss_(
                        epoch, args.alpha, checkpoint_average_stats, 'train'
                    )
                )
                checkpoint_stats = defaultdict(list)

            if global_step % args.valid_every == 0:
                with torch.no_grad():
                    results, output_ids, s_new = evaluate(
                        args, model, valid_dataloader, epoch
                    )

                    # Output evaluation statistics
                    logger.info(s_new)

                    # Add one hyperparameter target_acc to the yml file.
                    cur_acc = float(results['accuracy'])
                    if cur_acc > args.target_acc and cur_acc > best_acc:
                        best_acc = cur_acc
                        save(args, model, logger)
                        print(
                            'Epoch :', epoch, 'Prediction Accuracy =',
                            float(results['accuracy']), 'Saved to Path :',
                            args.output_dir
                        )
                        if args.TransferH:
                            args.hard = True


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
    args = parser.parse_args()
    args_dict = vars(args)
    with open(args_dict['config'], 'r') as config_file:
        args_dict.update(yaml.load(config_file))

    # set random seed
    set_seed(args)

    logging.info('Entering main run script')

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    train_captions = [
        json.loads(line) for line in open(args.train_captions, 'r').readlines()
    ]
    valid_captions = [
        json.loads(line) for line in open(args.valid_captions, 'r').readlines()
    ]
    train_images = torch.load(args.train_images)
    valid_images = torch.load(args.valid_images)

    logger.info('Dataset Loaded')

    # Write the model description
    logger.info('Configuration:')
    print(args)

    # Initialize agent
    model = CommunicationAgent(args)

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
    tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
    training_set = ImageCaptionDataset(
        train_images, train_captions, tokenizer, args
    )
    valid_set = ImageCaptionDataset(
        valid_images, valid_captions, tokenizer, args
    )
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
        model.load_state_dict(torch.load(checkpoint))
        model.to(args.device)
        model.eval()
        results, output_ids, s_new = evaluate(args, model, valid_dataloader)
        if args.save_output_txt:
            output_texts = ids_to_texts(output_ids, tokenizer)
            with open(args.output_dir + '/eval_texts.txt', 'w') as f:
                for i in output_texts:
                    f.write(i)

        logger.info("Best model stats: ")
        logger.info(s_new)


if __name__ == '__main__':
    main()