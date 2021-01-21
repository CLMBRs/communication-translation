import argparse
import codecs
import copy
import math
import os
import sys
import yaml
import numpy as np
import pickle as pkl
import subprocess as commands
from bart_models import *
from dataloader import *
from forward import *
from models import *
from util import *

import torch
import torch.autograd as autograd
import torch.nn as nn
from torch.autograd import Variable
from torchfile import load as load_lua

# General comments here:
# -Instead of using print, maybe we use logger?
# -Don't like the way how they define epoch, we should probably follow HF's
# style.
# -General model part is good, it is flexible and we could replace any model we
# want
# -We want to rewrite the data-loader part in the data_loader.py
# -We do not have a predict function yet


def main():
    # TODO: The seed should be a setable parameter
    random = np.random
    random.seed(42)

    parser = argparse.ArgumentParser(description='Ref Game Engine')
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    args_dict = vars(args)
    with open(args_dict["config"], "r") as config_file:
        args_dict.update(yaml.load(config_file))
        start_time = time.time()
        print("Entering Main")
    if args.dataset == "coco":
        feat_path = coco_path()
        data_path = coco_path()
        task_path = args.dataset
        args.l2 = "jp"
    else:
        print("image dataset should be set as coco")
        #here to insert alternative imgae data set
        #Xuhui: we should be able to use any img

    # Xuhui: this is loading the pre-computed ResNet Image representation
    (train_img1, train_img2, valid_img, test_img) = [
        torch.load(f'{feat_path}/half_feats/{x}') for x in
        (f"train_en_feats train_{args.l2}_feats valid_feats test_feats").split()
    ]

    print("Dataset Loaded")
    fixed, learned = [], ["lsn"]
    if args.fix_spk:  #False
        fixed.append("spk")
    else:
        learned.append("spk")

    if args.fix_bhd:  #False
        fixed.append("bhd")
    else:
        learned.append("bhd")

    fixed, learned = "_".join(sorted(fixed)), "_".join(sorted(learned))

    assert args.which_loss in "joint lsn".split()  #which_loss = 'joint'

    #fixed_.learned_bhd_lsn_spk.joint_loss/
    model_str = f"fixed_{fixed}.learned_{learned}.{args.which_loss}_loss/"
    if args.bart:
        model_str = "bart." + model_str
    if args.pretrain_spk:  #False
        model_str = "pretrain_spk." + model_str
    if args.no_share_bhd:  #False
        model_str = "no_share_bhd." + model_str

    mill = int(round(time.time() * 1000)) % 1000

    big = f"{saved_results_path()}sentence_level/{task_path}"
    path = f"{saved_results_path()}sentence_level/{task_path}/joint_model/"
    hyperparam_str = (
        f"{mill}_dropout_{args.dropout}.alpha_{args.alpha}.lr_{args.lr}"
        f".temp_{args.temp}.D_hid_{args.D_hid}.D_emb_{args.D_emb}"
        f".num_dist_{args.num_dist}.vocab_size_{args.vocab_size}"
        f"_{args.vocab_size}.hard_{args.hard}/"
    )
    path_dir = path + model_str + hyperparam_str
    # Xuhui: I actually do not like the way how they make the directory, any
    # better thoughts?
    if not args.no_write:
        recur_mkdir(path_dir)

    # sys.stdout = Logger(
    #     path_dir, no_write=args.no_write, no_terminal=args.no_terminal
    # )
    print(args)
    print(model_str)
    print(hyperparam_str)
    dir_dic = {
        "feat_path": feat_path,
        "data_path": data_path,
        "task_path": task_path,
        "path": path,
        "path_dir": path_dir
    }
    data = torch.cat([train_img1, train_img2, valid_img, test_img], dim=0)
    train_data, valid_data = remove_duplicate(data)
    train_data = train_data[:50000]
    print('train_img :', type(train_data), train_data.shape)
    print('valid_img :', type(valid_data), valid_data.shape)

    if args.bart:
        model = BartAgent(args)
    else:
        model = SingleAgent(args)

    print(model)
    if not args.cpu:
        torch.cuda.set_device(args.gpuid)
        model = model.cuda()

    #Xuhui: This module is showing the model parameter, maybe we do not need
    in_params, out_params = [], []
    in_names, out_names = [], []
    for name, param in model.named_parameters():
        if ("speaker" in name and args.fix_spk) or\
           ("beholder" in name and args.fix_bhd):
            out_params.append(param)
            out_names.append(name)
        else:
            in_params.append(param)
            in_names.append(name)

    in_size, out_size = [x.size()
                         for x in in_params], [x.size() for x in out_params]
    in_sum, out_sum = sum([np.prod(x) for x in in_size]), sum(
        [np.prod(x) for x in out_size]
    )

    print("IN    : {} params".format(in_sum))
    print("OUT   : {} params".format(out_sum))
    print("TOTAL : {} params".format(in_sum + out_sum))

    loss_fn = {
        'xent': nn.CrossEntropyLoss(),
        'mse': nn.MSELoss(),
        'mrl': nn.MarginRankingLoss(),
        'mlml': nn.MultiLabelMarginLoss(),
        'mml': nn.MultiMarginLoss()
    }
    tt = torch
    if not args.cpu:
        loss_fn = {k: v.cuda() for (k, v) in loss_fn.items()}
        tt = torch.cuda

    optimizer = torch.optim.Adam(in_params, lr=args.lr)

    best_epoch = -1
    train_loss_dict_ = get_log_loss_dict_()
    output_id_path = '/gscratch/ark/xuhuizh/UMT_datasentence_level/'
    for epoch in range(args.num_games):
        loss, _ = forward_joint(
            train_data, model, train_loss_dict_, args, loss_fn, args.num_dist,
            tt
        )
        optimizer.zero_grad()
        loss.backward()
        total_norm = nn.utils.clip_grad_norm(in_params, args.grad_clip)
        optimizer.step()

        if epoch % args.print_every == 0:
            avg_loss_dict_ = get_avg_from_loss_dict_(train_loss_dict_)
            print(print_loss_(epoch, args.alpha, avg_loss_dict_, "train"))
            train_loss_dict_ = get_log_loss_dict_()

        with torch.no_grad():
            model.eval()
            if epoch % args.valid_every == 0:
                valid_loss_dict_ = get_log_loss_dict_()
                output_ids = True
                for idx in range(args.print_every):
                    _, output_ids_batch = forward_joint(
                        valid_data, model, valid_loss_dict_, args, loss_fn,
                        args.num_dist_, tt
                    )
                if output_ids == True:
                    output_ids = output_ids_batch
                output_ids = torch.cat([output_ids, output_ids_batch], dim=0)
                avg_loss_dict_ = get_avg_from_loss_dict_(valid_loss_dict_)
                s_new = print_loss_(epoch, args.alpha, avg_loss_dict_, "valid")
                print(s_new)
                if float(s_new.split()[-6][:-2]) > 85.0:
                    path_model = path_dir + f"model_{float(s_new.split()[-6][:-2])}_{epoch}_{args.vocab_size}.pt"
                    torch.save(
                        output_ids, output_id_path + 'bart_output_ids.pt'
                    )
                    torch.save(model.state_dict(), path_model)
                    print(
                        "Epoch :", epoch, "Prediction Accuracy =",
                        float(s_new.split()[-6][:-2]), "Saved to Path :",
                        path_dir
                    )
                    if args.TransferH:
                        args.hard = True

        model.train()
    end_time = time.time()
    print("Total Runtime :", end_time - start_time)


if __name__ == '__main__':
    main()
