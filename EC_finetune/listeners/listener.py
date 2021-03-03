import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Module


class Listener(Module):
    """
    A "Listener" module for the encoder/receiver portion of a communication-game
    setup. Specific encoding stack is passed in as a parameter, and must meet
    certain interfaces (see Args)
    Args:
        encoder: A PyTorch module for encoding variable-length gumbel-generated
            messages to a hidden representation. Must have its embedding module
            accessible as ``encoder.embedding``. The output dimension of the
            encoder will effectively be the output dimension of the Listener
            module
        dropout: The dropout rate applied after the message is re-embedded
        unit_norm: Whether to divide the output by unit norm. Default: ``False``
    """
    def __init__(
        self, encoder: Module, dropout: float, unit_norm: bool = False
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.listener = encoder
        self.embedding = encoder.embedding
        self.unit_norm = unit_norm

    def forward(
        self,
        message_ids: Tensor = None,
        message_logits: Tensor = None,
        **kwargs
    ) -> Tensor:
        """
        Return the final listener representation of the input message
        Args:
            message_ids: The tensor of indices representing the input message as
                vocabulary items
            message_lengths: The tensor of message lengths (before padding)
            message_logits: The tensor of gumbel-generated distributions over
                vocabulary items, representing the input message
        """
        message_embedding = torch.matmul(message_logits, self.embedding.weight)
        message_embedding = self.dropout(message_embedding)
        output = self.listener(message_ids, message_embedding)

        if self.unit_norm:
            norm = torch.norm(output, p=2, dim=1, keepdim=True).detach() + 1e-9
            output = output / norm.expand_as(output)

        return output
