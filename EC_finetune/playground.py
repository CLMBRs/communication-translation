import argparse
import logging
import os
import random
import sys
import time
from collections import defaultdict
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartTokenizer
from tqdm import tqdm

from EC_finetune.agents import CommunicationAgent
from EC_finetune.dataloader import ImageIdentificationDataset, VisuaLingConstraintDataset
from EC_finetune.util import print_loss_, remove_duplicate


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, dataloader, epoch=0):
    stats = defaultdict(list)
    output_ids = True
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for batch in epoch_iterator:
        # Start evaluation mode
        model.eval()

        # Move data to the GPU
        batch['speaker_image'] = batch['speaker_image'].to(args.device)
        batch['listener_images'] = batch['listener_images'].to(args.device)
        batch['target'] = batch['target'].to(args.device)

        eval_return_dict = model(batch)

        # Xuhui: I do not understand this chunk of code
        '''
        if output_ids == True:
            output_ids = eval_return_dict['message']
            output_ids = False
        else:
            output_ids = torch.cat(
                [output_ids, eval_return_dict['message']], dim=0
            )
        '''

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
    torch.save(
        model.state_dict(), args.output_dir + '/model.pt'
    )
    # Good practice: save your training arguments together
    # with the trained model
    torch.save(
        args,
        os.path.join(args.output_dir, "training_args.bin")
    )

    # For pretrained models, provide extra saving strategy
    if args.save_pretrain_seperately:
        # Save a trained model, configuration and tokenizer
        # using `save_pretrained()`. They can then be
        # reloaded using `from_pretrained()`
        model_to_save = (
            model.model.module
            if hasattr(model, "module") else model.model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)


def train(args, model, dataloader, valid_dataloader, in_params, logger):
    optimizer = torch.optim.Adam(in_params, lr=args.lr)
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
            nn.utils.clip_grad_norm_(in_params, args.grad_clip)
            optimizer.step()
            global_step += 1
            if global_step >= args.max_global_step:
                # Save model even if it does not reach the target acc in the
                # end
                if best_acc ==0:
                    save(args, model, logger)
                return global_step

            train_return_dict['loss'] = loss.item()
            train_return_dict['mean_length'] = train_return_dict['mean_length'].item()

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
                        args, model, valid_dataloader, global_step
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
    return global_step


def main():
    """
    Pretrain multilingual model on the referential game and save it.
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
    parser = argparse.ArgumentParser(description='Ref Game Engine')
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    args_dict = vars(args)
    with open(args_dict['config'], 'r') as config_file:
        args_dict.update(yaml.load(config_file))

    # set random seed
    set_seed(args)

    # Start the clock for the beginning of the main function
    start_time = time.time()
    logging.info('Entering main run script')

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # TODO: allow use of other image datasets
    if args.dataset == 'coco':
        feat_path = args.coco_path
        data_path = args.coco_path
        task_path = args.dataset
        args.l2 = 'jp'
    else:
        # Here to insert alternative imgae data set
        # Xuhui: we should be able to use any img
        raise ValueError('image dataset should be set as coco')

    # Load the pre-computed ResNet Image representation
    # Xuhui: Consider moving the data loading process to the new MyDataset obj
    data_names = [
        'train_en_feats', f'train_{args.l2}_feats', 'valid_feats', 'test_feats'
    ]
    (train_img1, train_img2, valid_img, test_img) = [
        torch.load(f'{feat_path}/half_feats/{x}') for x in data_names
    ]

    logger.info('Dataset Loaded')

    # Write the model description

    logger.info('Configuration:')
    print(args)

    # Organize the data into a single tensor, remove duplicates, and trim to
    # the number of examples wanted
    data = torch.cat([train_img1, train_img2, valid_img, test_img], dim=0)
    train_data, valid_data = remove_duplicate(data)
    # TODO: This limit should be parameterized, not hard
    train_data = train_data[:50000]

    # Initialize agent
    model = CommunicationAgent(args)

    # Move the model to gpu if the configuration calls for it
    model.to(args.device)

    # Loop through the named parameters to find the number of input and output
    # parameters
    # Xuhui: This module is showing the model parameter, maybe we do not need
    in_params, out_params = [], []
    in_names, out_names = [], []
    for name, param in model.named_parameters():
        speaker_named = ('speaker' in name and args.fix_spk)
        beholder_named = ('beholder' in name and args.fix_bhd)
        if speaker_named or beholder_named:
            out_params.append(param)
            out_names.append(name)
        else:
            in_params.append(param)
            in_names.append(name)

    # Sum up the number of input and output parameters and log them
    in_size = [x.size() for x in in_params]
    out_size = [x.size() for x in out_params]
    in_sum = sum([np.prod(x) for x in in_size])
    out_sum = sum([np.prod(x) for x in out_size])
    logger.info(f'IN    : {in_sum} params')
    logger.info(f'OUT   : {out_sum} params')
    logger.info(f'TOTAL : {in_sum + out_sum} params')

    # Xuhui: Is this still necessary?
    loss_fn = {
        'xent': nn.CrossEntropyLoss(),
        'mse': nn.MSELoss(),
        'mrl': nn.MarginRankingLoss(),
        'mlml': nn.MultiLabelMarginLoss(),
        'mml': nn.MultiMarginLoss()
    }
    # Xuhui: This chunck of code seems redundant to me.
    if not args.cpu:
        loss_fn = {k: v.cuda() for (k, v) in loss_fn.items()}

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
    if args.model_name == 'mbart':
        tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
        training_set = VisuaLingConstraintDataset(
            train_data, args.num_distractors_train, args, tokenizer
        )
        valid_set = VisuaLingConstraintDataset(
            valid_data, args.num_distractors_valid, args, tokenizer
        )
    else:
        training_set = ImageIdentificationDataset(
            train_data, args.num_distractors_train
        )
        valid_set = ImageIdentificationDataset(
            valid_data, args.num_distractors_valid
        )
    training_dataloader = DataLoader(training_set, **training_params)
    valid_dataloader = DataLoader(valid_set, **test_params)

    if args.do_train:
        global_step = train(
            args, model, training_dataloader, valid_dataloader, in_params,
            logger
        )
    if args.do_eval:
        checkpoint = args.output_dir + '/model.pt'
        logger.info("Evaluate the following checkpoint: %s", checkpoint)
        model.load_state_dict(torch.load(checkpoint))
        model.to(args.device)
        model.eval()
        results, output_ids, s_new = evaluate(args, model, valid_dataloader)

    end_time = time.time()
    logger.info('Total Runtime :', end_time - start_time)


if __name__ == '__main__':
    main()
