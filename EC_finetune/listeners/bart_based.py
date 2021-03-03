import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Module

from EC_finetune.modelings.modeling_bart import BartForConditionalGeneration
from EC_finetune.modelings.modeling_mbart import MBartForConditionalGeneration


class BartEncoder(Module):
    """
    A Bart encoder module that fits the the interface necessary to be passed to
    the "Listener" class used in communication games
    Args:
        model: A BartForConditionalGeneration object to be used as the main
            encoder stack
        output_dim: The output dimension for the encoder/listener stack
    """
    def __init__(self, model: BartForConditionalGeneration, output_dim: int):
        super().__init__()
        self.embedding = model.model.shared
        self.encoder = model.gumbel_encoder
        self.hidden_to_output = nn.Linear(1024, output_dim)

    def forward(
        self, message_ids: Tensor, message_embedding: Tensor, **kwargs
    ) -> Tensor:
        """
        Return the pooled representation of the Bart encoder stack over the
        embedded input message
        Args:
            message_embedding: The tensor containing the input message, assumed
                to have been embedded by self.embedding, accessed by the
                Listener object to which this is passed
            message_lengths: The tensor of message lengths (before padding).
                Doesn't actually do anything here, but necessary to make the
                forward signature of the Listener encoders the same
        """
        hidden = self.encoder(input_ids=message_ids, input_embeds=message_embedding)
        pooled_hidden = torch.mean(hidden.last_hidden_state, dim=1)
        output = self.hidden_to_output(pooled_hidden)
        return output


class MBartEncoder(Module):
    def __init__(self, model: MBartForConditionalGeneration, output_dim: int):
        super().__init__()
        self.embedding = model.model.shared
        self.encoder = model.gumbel_encoder
        self.hidden_to_output = nn.Linear(1024, output_dim)

    def forward(
        self, message_ids: Tensor, message_embedding: Tensor, **kwargs
    ) -> Tensor:
        """
        Return the pooled representation of the Bart encoder stack over the
        embedded input message
        Args:
            message_embedding: The tensor containing the input message, assumed
                to have been embedded by self.embedding, accessed by the
                Listener object to which this is passed
            message_lengths: The tensor of message lengths (before padding).
                Doesn't actually do anything here, but necessary to make the
                forward signature of the Listener encoders the same
        """
        hidden = self.encoder(input_ids=message_ids, input_embeds=message_embedding)
        pooled_hidden = torch.mean(hidden.last_hidden_state, dim=1)
        output = self.hidden_to_output(pooled_hidden)
        return output


# TODO: this class should be deprecated; will delete once made sure
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
