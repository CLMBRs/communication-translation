import math
import operator
import os
import numpy as np
from .util import *

import torch
import torch.nn.functional as F
from torch.autograd import Variable


def sample_gumbel(shape, tt=torch, eps=1e-20):
    U = Variable(tt.FloatTensor(shape).uniform_(0, 1))
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax_sample(logits, temp, tt=torch, idx_=10):
    y = (logits + sample_gumbel(logits.size(), tt)) / temp
    if idx_ == 0:
        y[:, 3] = -float('inf')
    return F.softmax(y, dim=-1)


def gumbel_softmax(logits, temp, hard, tt=torch, idx_=10):
    # (batch_size, num_cat)
    y = gumbel_softmax_sample(logits, temp, tt, idx_)
    y_max, y_max_idx = torch.max(y, 1, keepdim=True)
    if hard:
        y_hard = tt.FloatTensor(y.size()).zero_().scatter_(1, y_max_idx.data, 1)
        y = Variable(y_hard - y.data, requires_grad=False) + y

    return y, y_max_idx
