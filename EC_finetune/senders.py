from abc import abstractmethod
from argparse import Namespace
from audioop import cross
from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .adapters import TransformerMapper
from .modelings.modeling_bart import _prepare_bart_decoder_inputs
from .modelings.modeling_mbart import MBartForConditionalGeneration


class Sender(Module):
    """
    Abstract class for the "sender" component of a Pytorch communication agent
    
    Senders implemented from various models can be passed to the
    CommunicationAgent constructor by instantiating this class. Python cannot
    impose input/output typing interfaces on classes, but subclases must
    implement the forward method and meet the interfaces described in its
    docstring for the CommunicationAgent to work
    """
    @abstractmethod
    def forward(self, sender_image: Tensor) -> Dict[str, Tensor]:
        """
        Convert a hidden representation of an image to a natural language
        sequence

        Subclasses of Sender must meet the Args and Returns interface
        constraints

        Args:
            sender_image: a Tensor batch of image representations for the
                Sender to caption/describe. `(batch_size, sender_image_dim)`
        Returns:
            a dictionary of Tensors with the following keys:
                `message_ids`: the sequence of output indices.
                    `(batch_size, max_seq_length)`
                `message_logits`: the output scores over the vocabulary.
                    `(batch_size, max_seq_length, vocab_size)`
                `message_lengths`: the length for each output. `(batch_size)`
        """
        raise NotImplementedError


class MBartSender(Sender):
    def __init__(
        self,
        model: MBartForConditionalGeneration,
        seq_len: int = None,
        unroll: str = None,
        unroll_length: int = 4,
        temperature: float = None,
        hard: bool = None,
        repetition_penalty: float = 1.0,
        beam_width: int = 1,
        generate_from_logits: bool = False,
        use_caption_crossattention: bool = False,
        img_ext_len: int = 10,
        tf_num_layers: int = 8
    ):
        """
        A Bart Sender subclass that can be input to a CommunicationAgent for
        communication games

        Args:
            model: a BartForConditionalGeneration object to be used as the main
                decoder stack
            seq_len: the maximum sequence length for the generations
            temperature: the distribution temperature for generation
            hard: whether generation should produce one-hot logits
        """
        super().__init__()
        self.top = model
        self.encoder = model.model.encoder
        self.decoder = model.model.decoder
        self.embedding = model.model.shared
        self.output_bias = model.final_logits_bias
        self.embedding_dim = model.model.shared.weight.size(1)
        self.unroll = unroll
        self.unroll_length = unroll_length
        self.top.temp = temperature
        self.top.hard = hard
        self.seq_len = seq_len
        self.repetition_penalty = repetition_penalty
        self.beam_width = beam_width
        self.generate_from_logits = generate_from_logits
        self.use_caption_crossattention = use_caption_crossattention
        self.img_ext_len = img_ext_len
        self.tf_num_layers = tf_num_layers

        if self.unroll == 'recurrent':
            print("Using 'recurrent' unrolling ")
            self.lstm = nn.LSTM(
                self.embedding_dim, self.embedding_dim, batch_first=True
            )
        elif self.unroll == 'transformer':
            print("Using 'transformer' unrolling")
            self.transformer = TransformerMapper(
                self.embedding_dim,
                self.embedding_dim,
                prefix_length=self.unroll_length,
                clip_length=self.img_ext_len,
                num_layers=self.tf_num_layers
            )
        else:
            print("No unrolling")

    def forward(
        self,
        sender_image: Tensor = None,
        sender_input_text: Tensor = None,
        sender_attention_mask: Tensor = None,
        decoder_input_ids: Tensor = None,
        **kwargs
    ):
        """
        Convert an image representation to a natural language message/caption.
        If decoder inputs are given, generate in the "supervised" manner by
        timestep. If no decoder inputs are given, freely gumbel-generate until
        the end of the sequence or max sequence length are reached

        The resulting message or caption is used by a receiver to pick out the
        original image from among distractors.

        Args:
            sender_image: batch of image representations for the Sender to
                caption/describe. `(batch_size, sender_image_dim)`
            decoder_input_ids: optional batch of decoder inputs for "supervised"
                generation. `(batch_size, max_seq_length)`. Default: `None`
            **kwargs: additional keyword arguments for gumbel generation. See
                `lang_id` and `lang_mask` below
        Returns:
            a dictionary of Tensors with the following keys (`message_lengths`
            only included when gumbel generating):
                `message_ids`: the sequence of output indices.
                    `(batch_size, max_seq_length)`
                `message_logits`: the output scores over the vocabulary.
                    `(batch_size, max_seq_length, vocab_size)`
                `message_lengths`: the length for each output. `(batch_size)`
        """
        device = self.embedding.weight.device

        # This is the input into gumbel generation; we will either unroll the
        # hidden image (either via recurrent unrolling or transformer unrolling)
        # or unroll text
        #
        # If decoder_input_ids is not None then we are performing caption
        # training and will skip the gumbel generation
        crossattention_input = None
        batch_size = None

        # Choose text vs caption-training vs image in the following order. Based
        # on config values, multiple conditions may be true during training. We
        # opt to enforce a strict ordering below to decide if EC should be
        # run on text then caption-training then images.
        text_unrolling_condition = sender_input_text is not None
        caption_training_condition = decoder_input_ids is not None
        image_unrolling_condition = sender_image is not None

        if text_unrolling_condition:
            batch_size = sender_input_text.size(0)
            sender_input_encodings = self.encoder(
                input_ids=sender_input_text,
                attention_mask=sender_attention_mask
            ).last_hidden_state
            crossattention_input = sender_input_encodings
        elif image_unrolling_condition:
            batch_size = sender_image.size(0)
            if len(sender_image.shape) == 2:
                sender_image = sender_image.unsqueeze(1)
            if self.unroll == 'recurrent':
                unrolled_hidden = []
                h = torch.zeros_like(sender_image).transpose(0, 1).to(
                    sender_image.device
                )
                c = torch.zeros_like(sender_image).transpose(0, 1).to(
                    sender_image.device
                )
                for i in range(self.unroll_length):
                    next_hidden, (h, c) = self.lstm(sender_image, (h, c))
                    unrolled_hidden.append(next_hidden)
                    sender_image = next_hidden
                sender_image = torch.stack(unrolled_hidden, dim=1).squeeze()
            elif self.unroll == 'transformer':
                sender_image = self.transformer(sender_image)
            crossattention_input = sender_image
        else:
            raise ValueError("Sender must receive valid input combination")

        # If decoder inputs are given, use them to generate timestep-wise
        if caption_training_condition:
            decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
                config=self.top.config,
                input_ids=decoder_input_ids,
                causal_mask_dtype=self.embedding.weight.dtype
            )
            output = self.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=crossattention_input,
                encoder_padding_mask=None,
                decoder_padding_mask=decoder_padding_mask,
                decoder_causal_mask=causal_mask
            )
            logits = F.linear(
                output.last_hidden_state,
                self.embedding.weight,
                bias=self.output_bias.to(device)
            )
            return {
                'message_ids': torch.argmax(logits, dim=2, keepdim=False),
                'message_logits': logits
            }

        # If no decoder inputs are given, use gumbel generation
        else:
            # Prepare special Bart parameters
            if 'lang_id' in kwargs:
                kwargs['lang_id'] = kwargs['lang_id'].view(batch_size, -1)
                kwargs['lang_id'] = \
                    kwargs['lang_id'].to(crossattention_input.device)
            if 'lang_mask' in kwargs:
                kwargs['lang_mask'] = \
                    kwargs['lang_mask'].to(crossattention_input.device)
            if self.generate_from_logits:
                kwargs['generate_from_logits'] = True

            # Get the sender model output and return
            output = self.top.gumbel_generate(
                input_embeds=crossattention_input,
                num_beams=self.beam_width,
                max_length=self.seq_len,
                repetition_penalty=self.repetition_penalty,
                no_repeat_ngram_size=4,
                **kwargs
            )
            return {
                'message_ids': output['generated_token_ids'],
                'message_logits': output['generated_logits'],
                'message_samples': output['generated_samples'],
                'message_lengths': output['generated_sentence_len']
            }


class RnnSender(Sender):
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
        self.sender = model
        self.input_dim = input_dim
        self.embedding_dim = self.sender.input_size
        self.hidden_dim = self.sender.hidden_size
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

    def forward(self, sender_image):
        """
        TODO: Add documentation
        """
        batch_size = sender_image.size(0)

        # Embed the image representation for use as the initial hidden state
        # for all layers of the RNN
        sender_image = sender_image.view(1, batch_size, self.input_dim)
        sender_image = self.projection(sender_image)
        sender_image = sender_image.repeat(self.num_layers, 1, 1)

        # Set the initial input for generation to be the BOS index, and run this
        # through the RNN to get the first output and hidden state
        initial_input = self.embedding(
            torch.ones([batch_size, 1]).cuda() * self.bos_idx
        )
        output, (h, c) = self.sender(initial_input, sender_image)

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
            output, (h, c) = self.sender(next_input, (h, c))
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
            'message_ids': message_ids,
            'message_logits': message_logits,
            'message_lengths': message_lengths
        }
