import torch.nn as nn
from torch.nn import Module

from EC_finetune.util import *


class RnnEncoder(Module):
    """
    An RNN encoder class that fits the interface necessary to be passed to the
    "Listener" class used in communication games
    Args:
        vocab_size: The size of the model vocabulary/embedding-table
        embedding_dim: The size of the encoder embedding
        output_dim: The output dimension for the encoder/listener stack
        num_layers: The number of RNN layers in the encoder. Default: ``1``
        bidirectional: Whether the RNN is bidirectional. Default: ``False``
        padding_idx: The index for the padding token. Default: ``0``
    """
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        output_dim: int,
        num_layers: int = 1,
        bidirectional: int = False,
        padding_idx: int = 0
    ):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=padding_idx
        )
        self.h_0 = nn.parameter.Parameter(
            torch.zeros(
                self.num_layers * self.num_directions, 1, self.output_dim
            )
        )
        self.rnn = nn.GRU(
            embedding_dim,
            output_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        num_directions = 2 if bidirectional else 1
        self.hidden_to_output = nn.Linear(
            num_directions * output_dim, output_dim
        )
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.num_directions = num_directions

    def forward(self, message_embedding, message_lengths):
        """
        Return the final representation of the RNN encoder stack over the
        embedded input message
        Args:
            message_embedding: The tensor containing the input message, assumed
                to have been embedded by self.embedding, accessed by the
                Listener object to which this is passed
            message_lengths: The tensor of message lengths (before padding)
        """
        batch_size = message_embedding.size(0)

        pack = torch.nn.utils.rnn.pack_padded_sequence(
            message_embedding,
            message_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, representation = self.rnn(pack, self.h_0.repeat(1, batch_size, 1))

        representation = representation[-self.num_directions:, :, :]
        output = representation.transpose(0, 1).contiguous().view(
            batch_size, self.num_directions * self.output_dim
        )
        output = self.hidden_to_output(output)
        return output
