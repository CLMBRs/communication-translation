import copy
import json
import operator
import time
import random
import numpy as np
import pickle as pkl
from collections import OrderedDict
from util import *

import torch
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torchfile import load as load_lua


class MyDataset(Dataset):
    def __init__(self, images, num_dist, tt):
        super(MyDataset, self).__init__()
        self.images = images
        self.img_index = list(range(len(images)))
        self.num_dist = num_dist
        self.tt = tt

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        dist_candidates = self.img_index[:index] + self.img_index[index+1:]
        dist_img = random.sample(dist_candidates, k = self.num_dist-1)

        spk_img = self.images[index]
        lsn_imgs = dist_img+[index]
        random.shuffle(lsn_imgs)

        which = lsn_imgs.index(index)
        lsn_imgs = torch.index_select(self.images, 0, torch.tensor(lsn_imgs)).numpy()

        return (spk_img, lsn_imgs, 0, 0, 0, 0, 0, which)


def next_batch_joint(images, batch_size, num_dist, tt):
    spk_imgs, spk_caps, lsn_imgs, lsn_caps, whichs = [], [], [], [], []
    total_indices = []
    keys = range(len(images))
    assert len(keys) >= num_dist
    for batch_idx in range(batch_size):
        img_indices = random.permutation(len(images))[:num_dist]
        # (1)
        which = random.randint(0, num_dist)
        spk_img = img_indices[which]
        # (batch_size, 2048)
        spk_imgs.append(spk_img)
        # batch_size * num_dist
        lsn_imgs += list(img_indices)
        # (batch_size)
        whichs.append(which)
    spk_imgs = torch.index_select(images, 0, torch.tensor(spk_imgs)).numpy()
    lsn_imgs = torch.index_select(images, 0, torch.tensor(lsn_imgs)).view(
        batch_size, num_dist, -1
    ).numpy()
    whichs = np.array(whichs)
    spk_imgs = Variable(torch.from_numpy(spk_imgs),
                        requires_grad=False).view(batch_size, -1)
    lsn_imgs = torch.from_numpy(lsn_imgs)
    lsn_imgs = Variable(lsn_imgs,
                        requires_grad=False).view(batch_size, num_dist, -1)

    whichs = Variable(torch.LongTensor(whichs),
                      requires_grad=False).view(batch_size)
    if tt == torch.cuda:
        spk_imgs = spk_imgs.cuda()
        lsn_imgs = lsn_imgs.cuda()
        whichs = whichs.cuda()
    return (spk_imgs, lsn_imgs, 0, 0, 0, 0, 0, whichs)


def weave_out(caps_out):
    ans = []
    seq_len = max([len(x) for x in caps_out])
    for idx in range(seq_len):
        for sublst in caps_out:
            if idx < len(sublst):
                ans.append(sublst[idx])
    return ans


