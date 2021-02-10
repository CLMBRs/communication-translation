import time
import operator
import math
import sys
import os
import pickle as pkl
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from EC_mbart_finetune.src.util import *
from EC_mbart_finetune.src.modeling_mbart import MBartForConditionalGeneration  # we make some customization
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

millis = int(round(time.time() * 1000))
torch.manual_seed(millis)
torch.cuda.manual_seed(millis)


class MBartAgent(torch.nn.Module):
    def __init__(self, args):
        super(MBartAgent, self).__init__()
        if args.no_share_bhd:
            print("Not sharing visual system for each agent.")
            self.beholder1 = Beholder(args)
            self.beholder2 = Beholder(args)
        else:
            print("Sharing visual system for each agent.")
            self.beholder = Beholder(args)
        # model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
        model = MBartForConditionalGeneration.from_pretrained('facebook/mbart-large-cc25')

        self.speaker = MBartSpeaker(model, args)
        self.listener = MBartListener(model, args)
        self.tt = torch if args.cpu else torch.cuda
        self.unit_norm = args.unit_norm

        self.beam_width = args.beam_width
        self.norm_pow = args.norm_pow
        self.no_share_bhd = args.no_share_bhd
        self.D_img = args.D_img
        self.D_hid = args.D_hid
        # if torch.cuda.is_available():
        #     self = self.cuda()

    def forward(self, data1, spk_sample_how):
        a_spk_img, b_lsn_imgs, a_spk_caps_in, a_spk_cap_lens, lang_ids, lang_masks = data1
        # spk_imgs : (batch_size, 2048)

        num_dist = b_lsn_imgs.size()[1]

        if self.no_share_bhd:
            spk_h_img = self.beholder1(a_spk_img)  # shared
        else:
            spk_h_img = self.beholder(a_spk_img)  # shared

        spk_msg, spk_embeds, spk_cap_len_ = self.speaker(spk_h_img, a_spk_caps_in, a_spk_cap_lens, lang_ids, lang_masks)

        lenlen = False
        if lenlen:
            print(spk_cap_len_[:10])
            end_idx = torch.max(torch.ones(spk_cap_len_.size()).cuda(),(spk_cap_len_-2).float())
            end_idx_ = torch.arange(0,end_idx.size(0)).cuda() * spk_logits.size(1)+end_idx.int()

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
        lis_hid = self.listener(spk_msg, spk_embeds)
        lis_hid = lis_hid.unsqueeze(1).repeat(1, num_dist, 1)  # (batch_size, num_dist, D_hid)

        return spk_embeds, (lis_hid,lsn_h_imgs), spk_msg, (end_idx_, end_loss_), (torch.min(spk_cap_len_.float()),
                                                                                  torch.mean(spk_cap_len_.float()),
                                                                                  torch.max(spk_cap_len_.float()))


class Beholder(torch.nn.Module):
    """
    This module is basically a linear module. It would receive an image or feautre of image and
    learn to coordinate the visual features with linguistic features with a few parameters added.
    """
    def __init__(self, args):
        super(Beholder, self).__init__()
        self.img_to_hid = torch.nn.Linear(args.D_img, args.D_hid)  # shared visual system
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


class MBartListener(torch.nn.Module):
    def __init__(self, mbart, args):
        super(MBartListener, self).__init__()
        self.hid_to_hid = nn.Linear(1024, args.D_hid)
        self.drop = nn.Dropout(p=args.dropout)

        self.D_hid = args.D_hid
        self.D_emb = args.D_emb
        self.num_layers = args.num_layers
        self.num_directions = args.num_directions
        self.vocab_size = args.vocab_size
        self.unit_norm = args.unit_norm

        self.tt = torch if args.cpu else torch.cuda
        self.lis = mbart.gumbel_encoder


    def forward(self, spk_msg, spk_msg_emb):
        # spk_msg : (batch_size, seq_len)
        # spk_msg_lens : (batch_size)
        batch_size = spk_msg.size()[0]
        seq_len = spk_msg.size()[1]

        spk_msg_emb = self.drop(spk_msg_emb)

        output = self.lis(input_ids=spk_msg, inputs_embeds=spk_msg_emb)
        # Mean pooling for now to match the img output
        output = torch.mean(output.last_hidden_state, dim=1)

        # Transform the dim to match the img dim

        out = self.hid_to_hid(output)

        return out


class MBartSpeaker(torch.nn.Module):
    def __init__(self, mbart, args):
        super(MBartSpeaker, self).__init__()
        self.spk = mbart
        self.project = nn.Linear(args.D_hid, args.seq_len * args.D_emb)

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

    def forward(self, img_hidden, caps_in, caps_in_lens, lang_ids, lang_masks):
        # here we assume lang_ids and masks are also batched and given
        # TODO: how to pack the input?

        batch_size = img_hidden.size()[0]  # caps_in.size()[0]
        assert batch_size == len(lang_ids) == len(lang_masks)

        img_hidden = self.project(img_hidden)
        img_hidden = img_hidden.view(-1, self.seq_len, self.D_emb)
        input_ids, input_embeds, cap_len = \
            self.spk.gumbel_generate(input_images=img_hidden, num_beams=1, max_length=self.seq_len, masks=lang_masks,
                                     decoder_input_ids=lang_ids.view(batch_size, -1))
        return input_ids, input_embeds, cap_len

