import torch.nn as nn

from EC_finetune.src.utils.util import *


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

    def forward(self, speaker_message, speaker_message_len, speaker_message_logits):
        # spk_msg : (batch_size, seq_len)
        # spk_msg_lens : (batch_size)
        batch_size = speaker_message.size()[0]
        seq_len = speaker_message.size()[1]

        h_0 = Variable(
            self.tt.FloatTensor(
                self.num_layers * self.num_directions, batch_size, self.D_hid
            ).zero_()
        )

        speaker_message_embedding = torch.matmul(speaker_message_logits, self.emb.weight)
        speaker_message_embedding = self.drop(speaker_message_embedding)

        pack = torch.nn.utils.rnn.pack_padded_sequence(
            speaker_message_embedding,
            speaker_message_len.cpu(),
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
