import torch.nn as nn
from torch.nn import Module

from EC_finetune.util import *


class RnnSpeaker(Module):
    def __init__(self, args):
        super().__init__()
        self.rnn = nn.GRU(
            args.embedding_dim,
            args.hidden_dim,
            args.num_layers,
            batch_first=True
        )
        self.embedding = nn.Embedding(
            args.vocab_size, args.embedding_dim, padding_idx=0
        )
        self.hiden_to_vocab = nn.Linear(args.hidden_dim, args.vocab_size)
        self.dropout = nn.Dropout(p=args.dropout)

        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.vocab_size = args.vocab_size
        self.bos_idx = args.bos_idx
        self.temp = args.temp
        self.hard = args.hard
        self.seq_len = args.seq_len

    def forward(self, h_image):
        batch_size = h_image.size(0)  # caps_in.size()[0]

        # Embed the image representation for use as the initial hidden state
        # for all layers of the RNN
        h_image = h_image.view(1, batch_size,
                               self.hidden_dim).repeat(self.num_layers, 1, 1)

        # Set the initial input for generation to be the BOS index, and run this
        # through the RNN to get the first output and hidden state
        initial_input = self.embedding(
            torch.ones([batch_size, 1]).cuda() * self.bos_idx
        )
        output, hidden = self.rnn(initial_input, h_image)

        # Loop through generation until the message is length seq_len. At each
        # step, get the logits over the vocab size, pass this through Gumbel
        # Softmax to get the generation distribution and the predicted vocab
        # id. Pass the predicted id to generate the next step
        logits = []
        labels = []
        for idx in range(self.seq_len):
            logit = self.hidden_to_vocab(output.view(-1, self.hidden_dim))
            logit_sample, label = F.gumbel_softmax(
                logit, tau=self.temp, hard=self.hard
            )
            next_input = torch.matmul(
                logit_sample.unsqueeze(1), self.embedding.weight
            )
            output, hidden = self.rnn(next_input, hidden)
            logits.append(logit_sample.unsqueeze(1))
            labels.append(label)
        logits = torch.cat(logits, dim=1)
        message_ids = torch.cat(labels, dim=-1)

        # Not sure what's going on with this block
        tmp = torch.zeros(logits.size(-1))
        tmp[3] = 1
        logits[:, -1, :] = tmp
        labels[:, -1] = 3
        pad_g = ((labels == 3).cumsum(1) == 0)
        labels = pad_g * labels
        pad_ = torch.zeros(logits.size()).cuda()
        pad_[:, :, 0] = 1
        message_logits = torch.where(
            pad_g.unsqueeze(-1).repeat(1, 1, logits.size(-1)), logits, pad_
        )
        message_lengths = pad_g.cumsum(1).max(1).values + 1

        return {
            "message_ids": message_ids,
            "message_logits": message_logits,
            "message_lengths": message_lengths
        }
