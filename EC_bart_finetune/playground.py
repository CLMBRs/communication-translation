import argparse
import logging
import os
import random
import sys
import time
import yaml
import numpy as np
from tqdm import tqdm

from agents import CommunicationAgent
from dataloader import ImageIdentificationDataset
from forward import forward_joint
from util import (
    get_log_loss_dict_, get_avg_from_loss_dict_, print_loss_, recur_mkdir,
    remove_duplicate
)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def evaluate(args, model, dataloader, loss_function, epoch):
    valid_loss_dict_ = get_log_loss_dict_()
    output_ids = True
    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for batch in epoch_iterator:
        # Start evaluation mode
        model.eval()

        # Move data to the GPU
        batch['speaker_image'] = batch['speaker_image'].to(args.device)
        batch['listener_images'] = batch['listener_images'].to(args.device)
        batch['target'] = batch['target'].to(args.device)

        _, output_ids_batch = forward_joint(
            batch, model, valid_loss_dict_, args, loss_function,
            args.num_distractors_train
        )

        if output_ids == True:
            output_ids = output_ids_batch
            output_ids = False
        output_ids = torch.cat([output_ids, output_ids_batch], dim=0)

    avg_loss_dict_ = get_avg_from_loss_dict_(valid_loss_dict_)
    s_new = print_loss_(epoch, args.alpha, avg_loss_dict_, 'valid')

    # AMD: I don't know what output_ids or s_new are, but it looks like you need
    # to return them in the evaluation loop
    return avg_loss_dict_, output_ids, s_new


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
    training_set = ImageIdentificationDataset(
        train_data, args.num_distractors_train
    )
    training_dataloader = DataLoader(training_set, **training_params)
    valid_set = ImageIdentificationDataset(
        valid_data, args.num_distractors_valid
    )
    valid_dataloader = DataLoader(valid_set, **test_params)

    optimizer = torch.optim.Adam(in_params, lr=args.lr)

    train_loss_dict_ = get_log_loss_dict_()
    # TODO: this path should be parameterized
    output_id_path = '/gscratch/ark/xuhuizh/UMT_datasentence_level/'
    for epoch in range(args.num_games):
        epoch_iterator = tqdm(training_dataloader, desc="Iteration")
        for batch in epoch_iterator:

            # Xuhui: Added this to inform the training started.
            model.train()

            # Xuhui: Added this to move data to the GPU
            batch['speaker_image'] = batch['speaker_image'].to(device)
            batch['listener_images'] = batch['listener_images'].to(device)
            batch['target'] = batch['target'].to(device)

            loss, _ = forward_joint(
                batch, model, train_loss_dict_, args, loss_fn,
                args.num_distractors_train
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(in_params, args.grad_clip)
            optimizer.step()

        if epoch % args.print_every == 0:
            avg_loss_dict_ = get_avg_from_loss_dict_(train_loss_dict_)
            logger.info(print_loss_(epoch, args.alpha, avg_loss_dict_, 'train'))
            train_loss_dict_ = get_log_loss_dict_()

        if epoch % args.valid_every == 0:
            with torch.no_grad():
                results, output_ids, s_new = evaluate(
                    args, model, valid_dataloader, loss_fn, epoch
                )

                # Output evaluation statistics
                logger.info(s_new)

                # Add one hyperparameter target_acc to the yml file.
                if float(results['accuracy']) > args.target_acc:
                    # Create output directory if needed
                    if not os.path.exists(args.output_dir):
                        os.makedirs(args.output_dir)

                    logger.info(
                        "Saving model checkpoint to %s", args.output_dir
                    )

                    if args.model == 'bart':
                        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
                        # They can then be reloaded using `from_pretrained()`
                        model_to_save = (
                            model.module if hasattr(model, "module") else model
                        )  # Take care of distributed/parallel training
                        model_to_save.save_pretrained(args.output_dir)

                        # Good practice: save your training arguments together with the trained model
                        torch.save(
                            args,
                            os.path.join(args.output_dir, "training_args.bin")
                        )

                    elif args.model == 'rnn':
                        torch.save(
                            model.state_dict(), args.output_dir + 'model_rnn.pt'
                        )
                    print(
                        'Epoch :', epoch, 'Prediction Accuracy =',
                        float(results['accuracy']), 'Saved to Path :',
                        args.output_dir
                    )
                    if args.TransferH:
                        args.hard = True

    end_time = time.time()
    logger.info('Total Runtime :', end_time - start_time)


if __name__ == '__main__':
    main()
