from abc import abstractmethod
from argparse import Namespace
from functools import reduce
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from .senders import Sender
from .receivers import Receiver


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split('.'))


class CommunicationAgent(Module):
    """
    An abstract PyTorch Module for creating a sender/receiver or
    speaker/listener agent set for communication games

    Sender and Receiver are passed in as arguments, and must meet the
    interfaces specified by the Sender and Receiver abstract classes,
    respectively. Beholder (image embedding compoment) can either be shared
    between Sender and Receiver, or independent for both
    Args:
        sender: the sender/speaker module, subclassed from Sender
        receiver: the receiver/listener module, subclassed from Receiver
        args: a dictionary of parameters
    """
    def __init__(self, sender: Sender, receiver: Receiver, args: Namespace):
        super().__init__()

        # Initialize the image Beholder, and clone if there is to be a separate
        # Beholder stack for both Sender and Receiver
        self.beholder = Beholder(
            image_dim=args.image_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
            unit_norm=args.unit_norm,
            two_ffwd=args.two_ffwd
        )
        if args.no_share_bhd:
            print("Not sharing visual system for each agent.")
            self.beholder1 = self.beholder
            self.beholder2 = Beholder(
                image_dim=args.image_dim,
                hidden_dim=args.hidden_dim,
                dropout=args.dropout,
                unit_norm=args.unit_norm,
                two_ffwd=args.two_ffwd
            )
        else:
            print("Sharing visual system for each agent.")
            self.beholder1 = self.beholder2 = self.beholder

        self.sender = sender
        self.receiver = receiver
        self.tokenizer = args.tokenizer

        self.image_dim = args.image_dim
        self.hidden_dim = args.hidden_dim
        self.unit_norm = args.unit_norm
        self.beam_width = args.beam_width
        self.no_share_bhd = args.no_share_bhd
        self.padding_index = args.padding_index
        self.cls_index = args.cls_index
        self.max_seq_length = args.max_seq_length

    def freeze_adapters(self) -> None:
        for param in self.beholder1.parameters():
            param.requires_grad = False
        for param in self.beholder2.parameters():
            param.requires_grad = False
        if self.sender.recurrent_unroll:
            for param in self.sender.lstm.parameters():
                param.requires_grad = False

    def image_to_message(self, batch: dict) -> dict:
        """
        Take in an image embedding and use the sender to return a
        message/caption representing the image

        If the batch contains `decoder_input_ids`, the sender generates in a
        "supervised" way (passing in the gold decoder input at each timestep).
        If not, Gumbel generation is used

        Args:
            batch: a dictionary of Tensors and other data. Must obligatorily
                include `sender_image`, the embedding of the image from which to
                generate the message. The rest of the batch is passed to the
                sender as kwargs
        """

        # Embed the Sender's image using the Beholder
        image_embedding = self.beholder1(batch['sender_image'])
        # Generate the Sender's message/caption about the image
        return self.sender(image_embedding, **batch)

    def choose_image_from_message(
        self, message_dict: dict, receiver_images: Tensor
    ) -> Tensor:
        """
        Take in a message from the sender, as well as a set of image choices,
        and use the receiver to embed/interpret the message. Compare the final
        receiver representation to the image embeddings to form logits over the
        choices

        Args:
            message_dict: a dictionary of Tensors representing the sender's
                message/caption. Must minimally include `message_ids`. See
                `EC_finetune.senders` for more information
            receiver_images: a Tensor of image embeddings for the model to
                choose between. `(batch_size, num_image_choices, image_dim)`
        Returns: a Tensor of logits over the image choices for the batch
        """
        num_image_choices = receiver_images.size(1)

        # Embed the Receiver's candidate images using the Beholder
        # (batch_size, num_image_choices, hidden_dim)
        receiver_image_embeddings = self.beholder2(receiver_images)

        # Encode the Sender's message to a hidden state to be used to select
        # a candidate image
        receiver_hidden = self.receiver(**message_dict)
        # (batch_size, num_image_choices, hidden_dim)
        receiver_hidden = receiver_hidden.unsqueeze(1).repeat(
            1, num_image_choices, 1
        )

        # Get the Mean Squared Error between the final receiver representation
        # and all of the candidate images
        image_candidate_errors = F.mse_loss(
            receiver_hidden, receiver_image_embeddings, reduction='none'
        ).mean(dim=2).squeeze(-1)

        # Transform this to the inverted MSE (for use as scores)
        image_candidate_logits = 1 / (image_candidate_errors + 1e-10)
        return image_candidate_logits

    @staticmethod
    def lang_id_to_end(input_ids, lengths, pad_index):
        device = input_ids.device
        buff = torch.full((input_ids.size(0), 1), pad_index, device=device)
        input_ids = torch.cat((input_ids, buff), dim=1)
        first_pad_indices = lengths.unsqueeze(-1)
        # yapf: disable
        new_ids = torch.scatter(
            input_ids, 1, first_pad_indices, input_ids
        )[:, 1:]
        # yapf: enable
        return new_ids

    @staticmethod
    def lang_id_logit_to_end(input_logits, lengths):
        device = input_logits.device
        hidden_size = input_logits.size(2)
        seq_length = input_logits.size(1) + 1

        # Since the longest sequence has no padding, add an extra pad position
        # to the end of dimension 1
        buff = torch.zeros(
            (input_logits.size(0), 1, hidden_size), device=device
        )
        input_logits = torch.cat((input_logits, buff), dim=1)

        # For scatter to be compatible with the backward pass, the `index`
        # argument needs to be the same size as the source. This block should
        # create the proper index to send the first element to the back of the
        # sequence (and sends the replaced element to the front)
        first_pad_indices = lengths.view(-1, 1, 1)
        first_pad_indices = first_pad_indices.repeat(1, seq_length, hidden_size)
        first_pad_indices[:, 1:, :] = torch.arange(1, seq_length).view(1, -1, 1)
        for idx, length in enumerate(lengths.tolist()):
            first_pad_indices[idx, length] = 0

        # Use `torch.scatter` to do the swap. Drop the first token which is now
        # just padding
        new_logits = torch.scatter(
            input_logits, 1, first_pad_indices, input_logits
        )[:, 1:]
        return new_logits

    @staticmethod
    def prepend_non_pad_to_mask(padding_mask):
        device = padding_mask.device
        batch_size = padding_mask.size(0)

        # Prepend an additional non-pad position to the padding mask
        non_pads = torch.ones((batch_size, 1), device=device)
        padding_mask = torch.cat((non_pads, padding_mask), dim=1)
        return padding_mask

    @staticmethod
    def prepend_cls(input_ids, cls_id):
        device = input_ids.device
        batch_size = input_ids.size(0)

        # Prepend the token acting as CLS to the beginning of the sequence
        cls_tokens = torch.full((batch_size, 1), cls_id, device=device)
        input_ids = torch.cat((cls_tokens, input_ids), dim=1)
        return input_ids

    @staticmethod
    def prepend_cls_logit(input_logits, cls_id):
        device = input_logits.device
        batch_size = input_logits.size(0)
        hidden_size = input_logits.size(2)

        # Form a one-hot representation of the CLS token and prepend it to the
        # beginning of the sequence of logits
        cls_onehot = (torch.arange(0, hidden_size) == cls_id).float()
        cls_onehot = cls_onehot.view(1, 1, -1).repeat(batch_size, 1, 1)
        cls_onehot = cls_onehot.to(device)
        return torch.cat((cls_onehot, input_logits), dim=1)

    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError


class ECImageIdentificationAgent(CommunicationAgent):
    def __init__(
        self,
        sender: Sender,
        receiver: Receiver,
        args: Namespace,
        language_model=None,
        orig_model=None
    ):
        super().__init__(sender, receiver, args)
        self.language_model_lambda = args.language_model_lambda
        self.weight_drift_lambda = args.weight_drift_lambda

        if self.language_model_lambda:
            if language_model is not None:
                self.language_model = language_model
            else:
                raise ValueError(
                    "A language model must be provided if"
                    " `language_model_lambda` is greater than 0.0"
                )

        if self.weight_drift_lambda:
            if orig_model is not None:
                self.orig_model = orig_model
            else:
                raise ValueError(
                    "A reference model must be provided if"
                    " `weight_drift_lambda` is greater than 0.0"
                )

    def forward(self, batch: dict) -> dict:
        """
        Train a model to convert the target image of a batch to a message using
        the sender and then encode the message back into a hidden state using
        the receiver. Use the final hidden state to choose the correct target
        image from among distractors

        Args:
            batch: a dictionary of Tensors and other data. Must obligatorily
                include `sender_image`, `receiver_images`, and `target`
        Returns:
            a dictionary of results with the following keys:
                `loss` (Tensor): the cross-entropy loss for selecting the
                    correct image
                `accuracy` (float): the percent of the image selections the
                    model chose correctly
                `message` (Tensor): the batch of messages as indices
                `mean_length` (float): the mean length of the messages
        """

        # Get the message dictionary (ids, logits, lengths) from the sender
        # based on the input image
        message_dict = self.image_to_message(batch)

        # Create the padding mask
        lengths = message_dict['message_lengths'].tolist()
        batch_size = len(lengths)
        max_length = min(max(lengths), self.max_seq_length)
        device = message_dict['message_ids'].device

        padding_mask = np.ones((batch_size, max_length))
        for seq in range(batch_size):
            padding_mask[seq][lengths[seq]:max_length] = 0
        padding_mask = torch.tensor(padding_mask)
        message_dict['attention_mask'] = padding_mask.to(device)

        if self.language_model_lambda:
            lm_ids = message_dict['message_ids']
            lm_logits = message_dict['message_logits']
            lm_padding_mask = message_dict['attention_mask']

        # Move the language id to the end of each sequence. We don't prepend
        # the CLS token now since the LM component will use the sequence
        # without the CLS as the targets
        message_dict['message_ids'] = self.lang_id_to_end(
            message_dict['message_ids'], message_dict['message_lengths'],
            self.padding_index
        )
        message_dict['message_logits'] = self.lang_id_logit_to_end(
            message_dict['message_logits'], message_dict['message_lengths']
        )

        # Optional language model loss block
        lm_loss = 0
        if self.language_model_lambda:
            lm_embeds = torch.matmul(
                lm_logits, self.language_model.shared.weight
            )
            lm_targets = message_dict['message_ids'].long()
            lm_padding_mask = lm_padding_mask.to(device)
            lm_logits = self.language_model(
                decoder_input_ids=lm_ids,
                input_embeds=lm_embeds,
                attention_mask=lm_padding_mask,
            )
            lm_loss = F.cross_entropy(
                lm_logits.transpose(1, 2),
                lm_targets.to(device),
                ignore_index=self.padding_index
            )
            lm_loss *= self.language_model_lambda

        # Prepend the CLS id to the beginning of the sequence, and adjust the
        # padding mask properly
        message_dict['attention_mask'] = self.prepend_non_pad_to_mask(
            message_dict['attention_mask']
        )
        message_dict['message_ids'] = self.prepend_cls(
            message_dict['message_ids'], self.cls_index
        )
        message_dict['message_logits'] = self.prepend_cls_logit(
            message_dict['message_logits'], self.cls_index
        )

        # Get the logits for the image choice candidates based on the sender's
        # message
        image_candidate_logits = self.choose_image_from_message(
            message_dict, batch['receiver_images']
        )

        # Get final cross-entropy loss between the candidates and the target
        # images
        target_image = batch['target']
        communication_loss = F.cross_entropy(
            image_candidate_logits, target_image
        )

        weight_drift_loss = 0
        if self.weight_drift_lambda:
            num_params = sum(
                [
                    p.numel()
                    for p in self.orig_model.parameters() if p.requires_grad
                ]
            )
            for key in dict(self.orig_model.named_parameters()).keys():
                weight_drift_loss += F.mse_loss(
                    rgetattr(self.sender.top, key),
                    rgetattr(self.orig_model, key),
                    reduction='sum'
                )
            if weight_drift_loss.item() > 0.0:
                weight_drift_loss = torch.pow(weight_drift_loss, 0.5)
            weight_drift_loss *= self.weight_drift_lambda

        overall_loss = communication_loss + lm_loss + weight_drift_loss

        # Get predicted accuracy
        _, predicted_idx = torch.max(image_candidate_logits, dim=1)
        eq = torch.eq(predicted_idx, target_image)
        accuracy = float(eq.sum().data) / float(eq.nelement())

        return_dict = {
            'loss': overall_loss,
            'communication loss': communication_loss.item(),
            'accuracy': 100 * accuracy,
            'message': message_dict['message_ids'],
            'mean_length': mean(lengths)
        }

        if self.language_model_lambda:
            return_dict.update(
                {'lm loss': lm_loss.item()}
            )
        if self.weight_drift_lambda:
            return_dict.update(
                {'drift loss': weight_drift_loss.item()}
            )
        return return_dict


class ImageCaptionGrounder(CommunicationAgent):
    """
    An agent to train grounding of text to images, by producing captions based
    on images, and picking images based on a caption (independently)
    """
    def __init__(self, sender: Sender, receiver: Receiver, args: Namespace):
        super().__init__(sender, receiver, args)
        if hasattr(args, 'image_selection_lambda'):
            self.image_selection_lambda = args.image_selection_lambda

    def forward(self, batch):
        """
        Train the sender to generate a gold-standard caption given an image.
        Also train to select the correct image from among distractors using the
        receiver's representation of the gold-standard caption

        Args:
            batch: a dictionary of Tensors and other data. Must obligatorily
                include `caption_ids`, `caption_mask`, `sender_image`,
                `receiver_images`, and `target`
        Returns:
            a dictionary of results with the following keys:
                `loss` (Tensor): the combination of the mean cross-entropy loss
                    for generating the caption and the cross-entropy for
                    selecting the correct image
                `caption generation loss` (float): the value of the loss for
                    caption generation
                `image selection loss` (float): the value of the loss for
                    selecting the correct image
                `accuracy` (float): the percent of the image selections the
                    model chose correctly
                `message` (Tensor): the batch of generated messages as indices
        """

        # Get the message dictionary (ids, logits) from the sender
        # based on the input image and calculate loss with the gold caption.
        # Adding the caption ids as `decoder_input_ids` puts the caption
        # generation into a supervised mode rather than free generation
        batch['decoder_input_ids'] = batch['caption_ids']
        message_dict = self.image_to_message(batch)
        caption_generation_loss = F.cross_entropy(
            message_dict['message_logits'].transpose(1, 2),
            batch['caption_ids'],
            ignore_index=self.padding_index
        )

        # Get the logits for the image choice candidates based on the gold
        # caption (NOT the sender's message)
        caption = {
            'message_ids': batch['caption_ids'],
            'attention_mask': batch['caption_mask'],
            'message_lengths': batch['caption_mask'].float().sum(dim=1).long()
        }

        # Prepend the CLS id to the beginning of the sequence, and adjust the
        # padding mask properly
        caption['attention_mask'] = self.prepend_non_pad_to_mask(
            caption['attention_mask']
        )
        caption['message_ids'] = self.prepend_cls(
            caption['message_ids'], self.cls_index
        )

        image_candidate_logits = self.choose_image_from_message(
            caption, batch['receiver_images']
        )
        # Get final cross-entropy loss between the candidates and the target
        # images
        target_image = batch['target']
        image_selection_loss = F.cross_entropy(
            image_candidate_logits, target_image
        )

        # Get predicted accuracy
        _, predicted_idx = torch.max(image_candidate_logits, dim=1)
        eq = torch.eq(predicted_idx, target_image)
        accuracy = float(eq.sum().data) / float(eq.nelement())

        if self.image_selection_lambda:
            image_selection_loss *= self.image_selection_lambda
        loss = caption_generation_loss + image_selection_loss

        return {
            'loss': loss,
            'caption generation loss': caption_generation_loss.item(),
            'image selection loss': image_selection_loss.item(),
            'accuracy': 100 * accuracy,
            'message': message_dict['message_ids']
        }


class Beholder(Module):
    """
    A "Beholder" module for embedding image data. Consists of one or two
    feedforward layers with optional dropout and unit norm
    Args:
        image_dim: The dimension of the input image embedding
        hidden_dim: The dimension of the hidden representation for the image
        dropout: The dropout rate after the first feedforward layer
        two_ffwd: Whether to use two feedforward layers. Default: ``False``
        unit_norm: Whether to divide the output by the unit norm. Default:
            ``False``
    """
    def __init__(
        self,
        image_dim: int,
        hidden_dim: int,
        dropout: float,
        unit_norm: bool = False,
        two_ffwd: bool = False
    ):
        super().__init__()
        self.image_to_hidden = torch.nn.Linear(image_dim, hidden_dim)
        self.unit_norm = unit_norm
        self.dropout = nn.Dropout(p=dropout)
        self.two_ffwd = two_ffwd
        if self.two_ffwd:
            self.hidden_to_hidden = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, image: Tensor) -> Tensor:
        """
        Return the embedding of the input image data
        Args:
            image: a Tensor embedding of image data
        Returns:
            h_image: The hidden representation of the image(s)
        """
        h_image = self.image_to_hidden(image)
        h_image = self.dropout(h_image)

        if self.two_ffwd:
            h_image = self.hidden_to_hidden(F.relu(h_image))

        if self.unit_norm:
            norm = torch.norm(h_image, p=2, dim=1, keepdim=True).detach() + 1e-9
            h_image = h_image / norm.expand_as(h_image)

        return h_image
