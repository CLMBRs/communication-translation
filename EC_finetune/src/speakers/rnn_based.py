import torch.nn as nn

from EC_finetune.src.util import *


class RnnSpeaker(torch.nn.Module):
    def __init__(self, lang, args):
        super(RnnSpeaker, self).__init__()
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

    def forward(self, speaker_images_hidden, **kwargs):
        # speaker_caps_in, speaker_cap_lens

        batch_size = speaker_images_hidden.size()[0]  # caps_in.size()[0]

        speaker_images_hidden = speaker_images_hidden.view(1, batch_size,
                                                           self.D_hid).repeat(self.num_layers, 1, 1)

        initial_input = self.emb(
            torch.ones([batch_size, 1], dtype=torch.int64).cuda() * 2
        )
        out_, hid_ = self.rnn(initial_input, speaker_images_hidden)
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
