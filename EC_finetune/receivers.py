from abc import abstractmethod

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Module

from .modelings.modeling_mbart import MBartForConditionalGeneration


class Receiver(Module):
    """
    Abstract class for the "receiver" component of a Pytorch communication agent
    
    Receivers implemented from various models can be passed to the
    CommunicationAgent constructor by instantiating this class. Python cannot
    impose input/output typing interfaces on classes, but subclases must
    implement the forward method and meet the interfaces described in its
    docstring for the CommunicationAgent to work
    """
    @abstractmethod
    def forward(
        self, message_ids: Tensor, message_logits: Tensor = None
    ) -> Tensor:
        """
        Convert a gumbel-generated sequence of natural language logits and ids
        to a hidden representation

        This hidden state is often used in EC games to pick out an image from
        distractors, but in principle it can serve any purpose. Subclasses of
        Receiver must meet the Args and Returns interface constraints

        Args:
            message_ids: the batch of generated sentences as indices.
                `(batch_size, sequence_length)`
            message_logits: the batch of generated sentences as logits over the
                vocabulary at each position.
                `(batch_size, sequence_length, vocab_size)`
        Returns:
            The batch of hidden states representing each sequence.
                `(batch_size, final_hidden_dim)`
        """
        raise NotImplementedError


class MBartReceiver(Receiver):
    """
    A Bart Receiver subclass that can be input to a CommunicationAgent for
    communication games

    Args:
        model: a BartForConditionalGeneration object to be used as the main
            encoder stack
        output_dim: the dimension of the final output representation
        dropout: the rate of dropout to be applied after the encoder. Default:
            `0.0`
        unit_norm: whether to unit-normalize the final output. Default: `False`
    """
    def __init__(
        self,
        model: MBartForConditionalGeneration,
        output_dim: int,
        recurrent_aggregation: bool = False,
        dropout: float = 0.0,
        unit_norm: bool = False
    ):
        super().__init__()
        self.embedding = model.model.shared
        self.encoder = model.model.encoder
        self.recurrent_aggregation = recurrent_aggregation
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.unit_norm = unit_norm

        if self.recurrent_aggregation:
            self.hidden_to_output = nn.LSTM(1024, output_dim, batch_first=True)
        else:
            self.hidden_to_output = nn.Linear(1024, output_dim)

    def forward(
        self,
        message_ids: Tensor,
        attention_mask: Tensor,
        message_lengths: Tensor,
        message_logits: Tensor = None
    ) -> Tensor:
        """
        Return the pooled representation of the Bart encoder stack over the
        embedded input message
        
        Args:
            message_ids: the batch of generated sentences as indices.
                `(batch_size, sequence_length)`
            attn_mask: the attention/padding mask for the batch.
                `(batch_size, sequence_length)`
            message_logits: the batch of generated sentences as logits over the
                vocabulary at each position. 
                `(batch_size, sequence_length, vocab_size)`
        Returns:
            The batch of hidden states representing each sequence.
                `(batch_size, output_dim)`
        """
        message_embedding = torch.matmul(
            message_logits, self.embedding.weight
        ) if message_logits is not None else self.embedding(message_ids)

        if self.dropout:
            message_embedding = self.dropout(message_embedding)
        hidden = self.encoder(
            input_ids=message_ids,
            input_embeds=message_embedding,
            attention_mask=attention_mask
        )
        hidden = hidden.last_hidden_state

        if self.recurrent_aggregation:
            _, (output, _) = self.hidden_to_output(hidden)
            output = output.squeeze()
        else:
            # hidden_size = hidden.size(2)
            # message_lengths = message_lengths - 1
            # length_indices = message_lengths.view(-1, 1, 1).repeat(1, 1, hidden_size)
            # classification_token = torch.gather(
            #     input=hidden, dim=1, index=length_indices
            # ).squeeze()

            # Use the initial CLS token as the sentence representation
            output = self.hidden_to_output(hidden[:, 0, :])
        return output


class RnnReceiver(Receiver):
    """
    An RNN Receiver subclass that can be input to a CommunicationAgent for
    communication games

    Args:
        model: an RNNBase object to be used as the main encoder stack
        vocab_size: the size of the natural language vocabulary
        output_dim: the dimension of the final output representation
        padding_idx: the index of the padding token
        dropout: the rate of dropout to be applied after the encoder. Default:
            `0.0`
        unit_norm: whether to unit-normalize the final output. Default: `False`
    """
    def __init__(
        self,
        model: nn.RNNBase,
        output_dim: int,
        vocab_size: int,
        padding_idx: int = 0,
        dropout: float = 0.0,
        unit_norm: bool = False
    ):
        super().__init__()
        self.encoder = model
        self.vocab_size = vocab_size
        self.embedding_dim = model.input_size
        self.hidden_dim = model.hidden_size
        self.output_dim = output_dim
        self.num_directions = 2 if model.bidirectional else 1
        self.num_layers = model.num_layers
        self.padding_index = padding_idx
        self.unit_norm = unit_norm

        self.embedding = nn.Embedding(
            vocab_size, self.embedding_dim, padding_idx=self.padding_index
        )
        self.h_0 = nn.parameter.Parameter(
            torch.zeros(
                self.num_layers * self.num_directions, 1, self.hidden_dim
            )
        )
        self.dropout = nn.Dropout(p=dropout) if dropout else None
        self.hidden_to_output = nn.Linear(
            self.num_directions * self.hidden_dim, self.output_dim
        )

    def forward(
        self,
        message_ids: Tensor,
        message_lengths: Tensor,
        message_logits: Tensor = None
    ):
        """
        Return the final representation of the RNN encoder stack over the
        embedded input message
        
        Args:
            message_ids: the batch of generated sentences as indices.
                `(batch_size, sequence_length)`
            message_logits: the batch of generated sentences as logits over the
                vocabulary at each position. 
                `(batch_size, sequence_length, vocab_size)`
            message_lengths: the batch of lengths for the generated sentences.
                `(batch_size)`
        Returns:
            The batch of hidden states representing each sequence.
                `(batch_size, output_dim)`
        """
        message_embedding = torch.matmul(
            message_logits, self.embedding.weight
        ) if message_logits is not None else self.embedding(message_ids)

        if self.dropout:
            message_embedding = self.dropout(message_embedding)
        packed_seq = torch.nn.utils.rnn.pack_padded_sequence(
            message_embedding,
            message_lengths,
            batch_first=True,
            enforce_sorted=False
        )
        batch_size = message_embedding.size(0)
        _, representation = self.rnn(
            packed_seq, self.h_0.repeat(1, batch_size, 1)
        )
        representation = representation[-self.num_directions:, :, :]
        concat_hidden = representation.transpose(0, 1).contiguous().view(
            batch_size, self.num_directions * self.hidden_dim
        )
        if self.unit_norm:
            norm = torch.norm(concat_hidden, p=2, dim=1, keepdim=True)
            norm = norm.detach() + 1e-9
            concat_hidden = concat_hidden / norm
        output = self.hidden_to_output(concat_hidden)
        return output
