import math
import operator
import os
import sys
import time
import numpy as np
import pickle as pkl
from util import *
from gumbel_utils import gumbel_softmax

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class SingleAgent(torch.nn.Module):
    def __init__(self, args):
        super(SingleAgent, self).__init__()
        if args.no_share_bhd:
            print("Not sharing visual system for each agent.")
            self.beholder1 = Beholder(args)
            self.beholder2 = Beholder(args)
        else:
            print("Sharing visual system for each agent.")
            self.beholder = Beholder(args)
        self.native, self.foreign = 'en', args.l2
        self.speaker = Speaker(self.native, args)
        self.listener = RnnListener(self.foreign, args)
        self.tt = torch if args.cpu else torch.cuda
        self.native, self.foreign = 'en', args.l2
        self.unit_norm = args.unit_norm

        self.beam_width = args.beam_width
        self.norm_pow = args.norm_pow
        self.no_share_bhd = args.no_share_bhd
        self.D_img = args.D_img
        self.D_hid = args.D_hid

    def forward(self, data1, spk_sample_how):
        # spk_imgs : (batch_size, 2048)
        a_spk_img, b_lsn_imgs, a_spk_caps_in, a_spk_cap_lens = data1

        num_dist = b_lsn_imgs.size()[1]

        if self.no_share_bhd:
            spk_h_img = self.beholder1(a_spk_img)  # shared
        else:
            spk_h_img = self.beholder(a_spk_img)  # shared

        spk_logits, comm_action, spk_cap_len_ = self.speaker(
            spk_h_img, a_spk_caps_in, a_spk_cap_lens, spk_sample_how
        )  # NOTE argmax / gumbel

        lenlen = False
        if lenlen:
            print(spk_cap_len_[:10])
            end_idx = torch.max(
                torch.ones(spk_cap_len_.size()).cuda(),
                (spk_cap_len_ - 2).float()
            )
            end_idx_ = torch.arange(0, end_idx.size(0)
                                   ).cuda() * spk_logits.size(1) + end_idx.int()

            end_loss_ = 3 * torch.ones(end_idx_.size()).long().cuda()
        else:
            end_idx_ = 0
            end_loss_ = 0

        lsn_imgs = b_lsn_imgs.view(-1, self.D_img)
        if self.no_share_bhd:
            lsn_h_imgs = self.beholder2(lsn_imgs)
        else:
            lsn_h_imgs = self.beholder(lsn_imgs)
        lsn_h_imgs = lsn_h_imgs.view(-1, num_dist, self.D_hid)
        rnn_hid = self.listener(
            comm_action[:, :-1], spk_cap_len_ - 1, spk_logits[:, :-1, :]
        )
        rnn_hid = rnn_hid.unsqueeze(1).repeat(
            1, num_dist, 1
        )  # (batch_size, num_dist, D_hid)

        return spk_logits, (rnn_hid,
                            lsn_h_imgs), comm_action, (end_idx_, end_loss_), (
                                torch.min(spk_cap_len_.float()),
                                torch.mean(spk_cap_len_.float()),
                                torch.max(spk_cap_len_.float())
                            )


class Beholder(torch.nn.Module):
    def __init__(self, args):
        super(Beholder, self).__init__()
        self.img_to_hid = torch.nn.Linear(
            args.D_img, args.D_hid
        )  # shared visual system
        self.unit_norm = args.unit_norm
        self.drop = nn.Dropout(p=args.dropout)
        self.two_fc = args.two_fc
        if self.two_fc:
            self.hid_to_hid = torch.nn.Linear(args.D_hid, args.D_hid)

    def forward(self, img):
        h_img = img

        h_img = self.img_to_hid(h_img)

        h_img = self.drop(h_img)

        if self.two_fc:
            h_img = self.hid_to_hid(F.relu(h_img))

        if self.unit_norm:
            norm = torch.norm(h_img, p=2, dim=1, keepdim=True).detach() + 1e-9
            h_img = h_img / norm.expand_as(h_img)
        return h_img


class RnnListener(torch.nn.Module):
    def __init__(self, lang, args):
        super(RnnListener, self).__init__()
        self.rnn = nn.GRU(args.D_emb, args.D_hid, args.num_layers, batch_first=True) if args.num_directions == 1 else \
                   nn.GRU(args.D_emb, args.D_hid, args.num_layers, batch_first=True, bidirectional=True)
        self.emb = nn.Embedding(args.vocab_size, args.D_emb, padding_idx=0)
        self.hid_to_hid = nn.Linear(
            args.num_directions * args.D_hid, args.D_hid
        )
        self.drop = nn.Dropout(p=args.dropout)

        self.D_hid = args.D_hid
        self.D_emb = args.D_emb
        self.num_layers = args.num_layers
        self.num_directions = args.num_directions
        self.vocab_size = args.vocab_size
        self.unit_norm = args.unit_norm

        self.tt = torch if args.cpu else torch.cuda

    def forward(self, spk_msg, spk_msg_lens, spk_logit=0):
        # spk_msg : (batch_size, seq_len)
        # spk_msg_lens : (batch_size)
        batch_size = spk_msg.size()[0]
        seq_len = spk_msg.size()[1]

        h_0 = Variable(
            self.tt.FloatTensor(
                self.num_layers * self.num_directions, batch_size, self.D_hid
            ).zero_()
        )

        spk_msg_emb = torch.matmul(spk_logit, self.emb.weight)
        spk_msg_emb = self.drop(spk_msg_emb)

        pack = torch.nn.utils.rnn.pack_padded_sequence(
            spk_msg_emb,
            spk_msg_lens.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, h_n = self.rnn(pack, h_0)

        h_n = h_n[-self.num_directions:, :, :]
        out = h_n.transpose(0, 1).contiguous().view(
            batch_size, self.num_directions * self.D_hid
        )
        out = self.hid_to_hid(out)

        if self.unit_norm:
            norm = torch.norm(out, p=2, dim=1, keepdim=True).detach() + 1e-9
            out = out / norm.expand_as(out)

        return out


class RnnSpeaker(torch.nn.Module):
    def __init__(self, lang, args):
        super(Speaker, self).__init__()
        self.rnn = nn.GRU(
            args.D_emb, args.D_hid, args.num_layers, batch_first=True
        )
        self.emb = nn.Embedding(args.vocab_size, args.D_emb, padding_idx=0)

        self.hid_to_voc = nn.Linear(args.D_hid, args.vocab_size)

        self.D_emb = args.D_emb
        self.D_hid = args.D_hid
        self.num_layers = args.num_layers
        self.drop = nn.Dropout(p=args.dropout)

        self.vocab_size = args.vocab_size
        self.temp = args.temp
        self.hard = args.hard
        self.tt = torch if args.cpu else torch.cuda
        self.tt_ = torch
        self.seq_len = args.seq_len

    def forward(self, h_img, caps_in, caps_in_lens, sample_how):

        batch_size = h_img.size()[0]  # caps_in.size()[0]

        h_img = h_img.view(1, batch_size,
                           self.D_hid).repeat(self.num_layers, 1, 1)

        initial_input = self.emb(
            torch.ones([batch_size, 1], dtype=torch.int64).cuda() * 2
        )
        out_, hid_ = self.rnn(initial_input, h_img)
        logits_ = []
        labels_ = []
        for idx in range(self.seq_len):
            logit_ = self.hid_to_voc(out_.view(-1, self.D_hid))
            c_logit_, comm_label_ = gumbel_softmax(
                logit_, self.temp, self.hard, self.tt, idx
            )

            input_ = torch.matmul(c_logit_.unsqueeze(1), self.emb.weight)
            out_, hid_ = self.rnn(input_, hid_)
            logits_.append(c_logit_.unsqueeze(1))
            labels_.append(comm_label_)
        logits_ = torch.cat(logits_, dim=1)
        labels_ = torch.cat(labels_, dim=-1)
        tmp = torch.zeros(logits_.size(-1))
        tmp[3] = 1
        logits_[:, -1, :] = tmp
        labels_[:, -1] = 3
        pad_g = ((labels_ == 3).cumsum(1) == 0)
        labels_ = pad_g * labels_
        pad_ = torch.zeros(logits_.size()).cuda()
        pad_[:, :, 0] = 1
        logits_ = torch.where(
            pad_g.unsqueeze(-1).repeat(1, 1, logits_.size(-1)), logits_, pad_
        )

        cap_len = pad_g.cumsum(1).max(1).values + 1

        return logits_, labels_, cap_len
