import torch
import torch.nn as nn


class BartListener(torch.nn.Module):
    def __init__(self, bart, lang, args):
        super(BartListener, self).__init__()
        self.hid_to_hid = nn.Linear(1024, args.D_hid)
        self.drop = nn.Dropout(p=args.dropout)

        self.D_hid = args.D_hid
        self.D_emb = args.D_emb
        self.num_layers = args.num_layers
        self.num_directions = args.num_directions
        self.vocab_size = args.vocab_size
        self.unit_norm = args.unit_norm

        self.tt = torch if args.cpu else torch.cuda
        self.lis = bart.gumbel_encoder
        self.emb = bart.model.shared

    def forward(self, speaker_message, speaker_message_logits, **kwargs):
        # spk_msg : (batch_size, seq_len)
        # spk_msg_lens : (batch_size)
        batch_size = speaker_message.size()[0]
        speaker_message_embedding = torch.matmul(speaker_message_logits, self.emb.weight)
        speaker_message_embedding = self.drop(speaker_message_embedding)

        output = self.lis(speaker_message, speaker_message_embedding)
        # Mean pooling for now to match the img output
        output = torch.mean(output.last_hidden_state, dim=1)

        # Transform the dim to match the img dim
        out = self.hid_to_hid(output)

        return out


class MBartListener(torch.nn.Module):
    def __init__(self, mbart, lang, args):
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
        self.emb = mbart.model.shared

    def forward(self, speaker_message, speaker_message_logits, **kwargs):
        # spk_msg : (batch_size, seq_len)
        # spk_msg_lens : (batch_size)
        batch_size = speaker_message.size()[0]
        # TODO: modify the modeling module
        speaker_message_embedding = torch.matmul(speaker_message_logits, self.emb.weight)
        speaker_message_embedding = self.drop(speaker_message_embedding)

        speaker_message_embedding = self.drop(speaker_message_embedding)

        output = self.lis(input_ids=speaker_message, inputs_embeds=speaker_message_embedding)
        # Mean pooling for now to match the img output
        output = torch.mean(output.last_hidden_state, dim=1)

        # Transform the dim to match the img dim

        out = self.hid_to_hid(output)

        return out
