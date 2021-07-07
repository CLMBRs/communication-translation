from abc import abstractmethod
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from EC_finetune.modelings.modeling_bart import _prepare_bart_decoder_inputs


class Speaker(Module):
    """
    Abstract class for the "speaker" component of a Pytorch communication agent
    
    Speakers implemented from various models can be passed to the
    CommunicationAgent constructor by instantiating this class. Python cannot
    impose input/output typing interfaces on classes, but subclases must
    implement the forward method and meet the interfaces described in its
    docstring for the CommunicationAgent to work
    """
    @abstractmethod
    def forward(self, image_hidden: Tensor) -> Dict[str, Tensor]:
        """
        Convert a hidden representation of an image to a natural language
        sequence

        Subclasses of Speaker must meet the Args and Returns interface
        constraints

        Args:
            image_hidden: a Tensor batch of image representations for the
                Speaker to caption/describe. `(batch_size, image_hidden_dim)`
        Returns:
            a dictionary of Tensors with the following keys:
                `message_ids`: the sequence of output indices.
                    `(batch_size, max_output_seq_length)`
                `message_logits`: the output scores over the vocabulary.
                    `(batch_size, max_output_seq_length, vocab_size)`
                `message_lengths`: the length for each output. `(batch_size)`
        """
        raise NotImplementedError


class BartSpeaker(Speaker):
    def __init__(
        self, model, input_dim, seq_len=None, temperature=None, hard=None
    ):
        """
        TODO: Add documentation
        """
        super().__init__()
        self.speaker = model
        self.decoder = model.model.decoder
        self.input_dim = input_dim
        self.embedding_dim = model.model.shared.weight.size(1)
        self.speaker.temp = temperature
        self.speaker.hard = hard
        self.seq_len = seq_len

        self.projection = nn.Linear(self.input_dim, self.embedding_dim)

    def forward(
        self, image_hidden: Tensor, decoder_input_ids: Tensor = None, **kwargs
    ):
        """
        TODO: Add documentation
        """
        # Ensure the batch is the correct shape
        # (batch_size, image_hidden_dim)
        batch_size = image_hidden.size(0)
        assert len(image_hidden.shape) == 2
        # (batch_size, max_sequence_length, image_hidden_dim)
        image_hidden = image_hidden.unsqueeze(1).repeat(1, self.seq_len, 1)

        # Embed the image hidden states to the speaker's embedding dim
        # (batch_size, max_sequence_length, embedding_dim)
        image_hidden = self.projection(image_hidden)

        if decoder_input_ids is not None:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                config=self.speaker.config,
                input_ids=decoder_input_ids,
                causal_mask_dtype=self.speaker.model.shared.weight.dtype
            )
            output = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=image_hidden,
                encoder_padding_mask=None,
                decoder_padding_mask=decoder_padding_mask,
                decoder_causal_mask=causal_mask
            )
            logits = F.linear(
                output.last_hidden_state,
                self.speaker.model.shared.weight,
                bias=self.speaker.final_logits_bias
            )
            return {
                "message_ids": torch.argmax(logits, dim=2, keepdim=False),
                "message_logits": logits
            }
        else:
            # Prepare special Bart parameters
            if "lang_id" in kwargs:
                kwargs["lang_id"] = kwargs["lang_id"].view(batch_size, -1)
                kwargs["lang_id"] = \
                    kwargs["lang_id"].to(image_hidden.device)
            if "lang_mask" in kwargs:
                kwargs["lang_mask"] = \
                    kwargs["lang_mask"].to(image_hidden.device)

            # Get the speaker model output and return
            output = self.speaker.gumbel_generate(
                input_images=image_hidden,
                num_beams=1,
                max_length=self.seq_len,
                **kwargs
            )
            return {
                "message_ids": output["generated_token_ids"],
                "message_logits": output["generated_logits"],
                "message_lengths": output["generated_sentence_len"]
            }


MBartSpeaker = BartSpeaker


class RnnSpeaker(Speaker):
    def __init__(
        self,
        model: nn.RNNBase,
        input_dim: int,
        vocab_size: int,
        bos_idx: int,
        dropout: float = 0.0,
        output_length: int = 256,
        padding_idx: int = 0,
        temperature: float = None,
        hard: bool = None
    ):
        """
        TODO: Add documentation
        """
        self.speaker = model
        self.input_dim = input_dim
        self.embedding_dim = self.speaker.input_size
        self.hidden_dim = self.speaker.hidden_size
        self.vocab_size = vocab_size

        self.projection = nn.Linear(input_dim, self.embedding_dim)
        self.embedding = nn.Embedding(
            self.vocab_size, self.embedding_dim, padding_idx=padding_idx
        )
        self.hiden_to_vocab = nn.Linear(self.hidden_dim, self.vocab_size)
        self.dropout = nn.Dropout(p=dropout)

        self.bos_idx = bos_idx
        self.temperature = temperature
        self.hard = hard
        self.output_length = output_length

    def forward(self, image_hidden):
        """
        TODO: Add documentation
        """
        batch_size = image_hidden.size(0)

        # Embed the image representation for use as the initial hidden state
        # for all layers of the RNN
        image_hidden = image_hidden.view(1, batch_size, self.input_dim)
        image_hidden = self.projection(image_hidden)
        image_hidden = image_hidden.repeat(self.num_layers, 1, 1)

        # Set the initial input for generation to be the BOS index, and run this
        # through the RNN to get the first output and hidden state
        initial_input = self.embedding(
            torch.ones([batch_size, 1]).cuda() * self.bos_idx
        )
        output, (h, c) = self.speaker(initial_input, image_hidden)

        # Loop through generation until the message is length seq_len. At each
        # step, get the logits over the vocab size, pass this through Gumbel
        # Softmax to get the generation distribution and the predicted vocab
        # id. Pass the predicted id to generate the next step
        logits = []
        labels = []
        for idx in range(self.seq_len):
            logit = self.hidden_to_vocab(output.view(-1, self.hidden_dim))
            logit_sample, label = F.gumbel_softmax(
                logit, tau=self.temperature, hard=self.hard
            )
            next_input = torch.matmul(
                logit_sample.unsqueeze(1), self.embedding.weight
            )
            output, (h, c) = self.speaker(next_input, (h, c))
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
