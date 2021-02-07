import argparse
import codecs
import copy
import logging
import math
import os
import random
import sys
import yaml
import numpy as np
import pickle as pkl
import subprocess as commands
from tqdm import tqdm, trange

from bart_models import *
from dataloader import *
from forward import *
from models import *
from util import *

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchfile import load as load_lua

# General comments here (Xuhui):
# -Don't like the way how they define epoch, we should probably follow HF's
# style.
# -General model part is good, it is flexible and we could replace any model we
# want
# -We want to rewrite the data-loader part in the data_loader.py
# -We do not have a predict function yet


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    """
    TODO: add a real docstring
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

    # TODO: The seed should be a setable parameter
    # set random seed
    set_seed(args)

    # Xuhui: Do we really need this?
    # Start the clock for the beginning of the main function
    start_time = time.time()
    logging.info('Entering main run script')

    # Setup CUDA, GPU  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # TODO: allow use of other image datasets
    if args.dataset == 'coco':
        feat_path = coco_path()
        data_path = coco_path()
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
    # TODO: what on earth are these string arguments? This needs to be a lot
    # more clear
    fixed, learned = [], ['lsn']
    if args.fix_spk:  #False
        fixed.append('spk')
    else:
        learned.append('spk')

    if args.fix_bhd:  #False
        fixed.append('bhd')
    else:
        learned.append('bhd')
    fixed, learned = '_'.join(sorted(fixed)), '_'.join(sorted(learned))

    assert args.which_loss in ['joint', 'lsn']
    model_str = f'fixed_{fixed}.learned_{learned}.{args.which_loss}_loss/'
    if args.bart:
        model_str = 'bart.' + model_str
    if args.pretrain_spk:
        model_str = 'pretrain_spk.' + model_str
    if args.no_share_bhd:
        model_str = 'no_share_bhd.' + model_str

    # Write the hyperparameter description
    # TODO: Why does this need to be put in the hyperparameter discription? If
    # it's for defining a random seed, we really need to replace it with a
    # single parameterized seed
    mill = int(round(time.time() * 1000)) % 1000
    big = f'{saved_results_path()}sentence_level/{task_path}'
    path = f'{saved_results_path()}sentence_level/{task_path}/joint_model/'
    hyperparam_str = (
        f'{mill}_dropout_{args.dropout}.alpha_{args.alpha}.lr_{args.lr}'
        f'.temp_{args.temp}.D_hid_{args.D_hid}.D_emb_{args.D_emb}'
        f'.num_dist_{args.num_dist}.vocab_size_{args.vocab_size}'
        f'_{args.vocab_size}.hard_{args.hard}/'
    )

    # Make the path to the model
    path_dir = path + model_str + hyperparam_str
    # Xuhui: I actually do not like the way how they make the directory, any
    # better thoughts?
    if not args.no_write:
        recur_mkdir(path_dir)

    # Log the general model information

    logger.info('Configuration:')
    print(args)
    logger.info('Model Name:')
    print(model_str)
    logger.info('Hyperparameters:')
    print(hyperparam_str)
    dir_dic = {
        'feat_path': feat_path,
        'data_path': data_path,
        'task_path': task_path,
        'path': path,
        'path_dir': path_dir
    }


    # Organize the data into a single tensor, remove duplicates, and trim to
    # the number of examples wanted
    data = torch.cat([train_img1, train_img2, valid_img, test_img], dim=0)
    train_data, valid_data = remove_duplicate(data)
    # TODO: This limit should be parameterized, not hard
    train_data = train_data[:50000]
    #logger.info('train_img :', type(train_data), train_data.shape)
    #logger.info('valid_img :', type(valid_data), valid_data.shape)

    # TODO: Choose between Bart or... what else?
    # Xuhui: The default is RNN
    if args.bart:
        model = BartAgent(args)
    else:
        model = SingleAgent(args)

    #logger.info("Model Info:")
    #print(model)

    # Move the model to gpu if the configuration calls for it
    # TODO: this should also probably check cuda.is_available()
    if args.n_gpu>0:
        torch.cuda.set_device(args.gpuid)
        model = model.cuda()

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
    tt = torch
    if not args.cpu:
        loss_fn = {k: v.cuda() for (k, v) in loss_fn.items()}
        tt = torch.cuda

    # Initialize the dataloader 
    training_params = {"batch_size": args.batch_size,
                                              "shuffle": True,
                                              "drop_last": True}
    test_params = {"batch_size": args.batch_size,
                                      "shuffle": False,
                                      "drop_last": False}
    training_set = MyDataset(train_data, args.num_dist, tt)
    training_generator = DataLoader(training_set, **training_params)
    valid_set = MyDataset(valid_data, args.num_dist_, tt)
    valid_generator = DataLoader(valid_set, **test_params)

    optimizer = torch.optim.Adam(in_params, lr=args.lr)

    best_epoch = -1
    train_loss_dict_ = get_log_loss_dict_()
    # TODO: this path should be parameterized
    output_id_path = '/gscratch/ark/xuhuizh/UMT_datasentence_level/'
    for epoch in range(args.num_games):
        epoch_iterator = tqdm(training_generator, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):

            # Xuhui: Added this to inform the training started. 
            model.train()

            # Xuhui: Added this to move data to the GPU
            batch = tuple(t.to(args.device) for t in batch)

            loss = forward_joint(
                batch, model, train_loss_dict_, args, loss_fn, args.num_dist,
                tt
            )
            optimizer.zero_grad()
            loss.backward()
            total_norm = nn.utils.clip_grad_norm(in_params, args.grad_clip)
            optimizer.step()

        if epoch % args.print_every == 0:
            avg_loss_dict_ = get_avg_from_loss_dict_(train_loss_dict_)
            logger.info(print_loss_(epoch, args.alpha, avg_loss_dict_, 'train'))
            train_loss_dict_ = get_log_loss_dict_()

        # TODO: I think that the if epoch %... conditional should come first
        with torch.no_grad():
            if epoch % args.valid_every == 0:
                valid_loss_dict_ = get_log_loss_dict_()
                output_ids = True
                for idx in range(args.print_every):
                    utput_l2[0]
                    _, output_ids_batch = forward_joint(
                        valid_data, model, valid_loss_dict_, args, loss_fn,
                        args.num_dist_, tt
                    )
                if output_ids == True:
                    output_ids = output_ids_batch
                output_ids = torch.cat([output_ids, output_ids_batch], dim=0)
                avg_loss_dict_ = get_avg_from_loss_dict_(valid_loss_dict_)
                s_new = print_loss_(epoch, args.alpha, avg_loss_dict_, 'valid')
                logger.info(s_new)
                if float(s_new.split()[-6][:-2]) > 85.0:
                    path_model = path_dir + f'model_{float(s_new.split()[-6][:-2])}_{epoch}_{args.vocab_size}.pt'
                    torch.save(
                        output_ids, output_id_path + 'bart_output_ids.pt'
                    )
                    torch.save(model.state_dict(), path_model)
                    print(
                        'Epoch :', epoch, 'Prediction Accuracy =',
                        float(s_new.split()[-6][:-2]), 'Saved to Path :',
                        path_dir
                    )
                    if args.TransferH:
                        args.hard = True

    end_time = time.time()
    logger.info('Total Runtime :', end_time - start_time)


if __name__ == '__main__':
    main()
