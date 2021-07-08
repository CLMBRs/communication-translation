from abc import abstractmethod
from argparse import Namespace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from EC_finetune.senders import Sender
from EC_finetune.receivers import Receiver


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
            self.beholder2 = self.beholder1.clone()
        else:
            print("Sharing visual system for each agent.")
            self.beholder1 = self.beholder2 = self.beholder

        self.sender = sender
        self.receiver = receiver

        self.image_dim = args.image_dim
        self.hidden_dim = args.hidden_dim
        self.unit_norm = args.unit_norm
        self.beam_width = args.beam_width
        self.no_share_bhd = args.no_share_bhd
        self.padding_index = args.padding_index
        self.max_seq_length = args.max_seq_length

    def image_to_message(self, batch):
        # Embed the Sender's image using the Beholder
        image_embedding = self.beholder1(batch['sender_image'])
        # Generate the Sender's message/caption about the image
        return self.sender(image_embedding, **batch)

    def choose_image_from_message(self, message_dict, receiver_images):
        num_image_choices = receiver_images.size(1)

        # Embed the Receiver's candidate images using the Beholder
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

    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError


class ECImageIdentificationAgent(CommunicationAgent):
    def forward(self, batch):
        # Get the message dictionary (ids, logits, lengths) from the sender
        # based on the input image
        message_dict = self.image_to_message(batch)

        # Create the padding mask
        lengths = list(message_dict['message_lengths'])
        batch_size = len(lengths)
        padding_mask = np.ones(batch_size, self.max_seq_length)
        for seq in range(batch_size):
            padding_mask[seq][lengths[seq]:self.max_seq_length] = 0
        padding_mask = torch.tensor(padding_mask).to(
            message_dict['message_ids'].device
        )
        message_dict['attention_mask'] = padding_mask

        # Get the logits for the image choice candidates based on the sender's
        # message
        image_candidate_logits = self.choose_image_from_message(
            message_dict, batch['receiver_images']
        )

        # Get final cross-entropy loss between the candidates and the target
        # images
        target_image = batch['target']
        communication_loss = self.crossentropy(
            image_candidate_logits, target_image
        )

        # Get predicted accuracy
        _, predicted_idx = torch.max(image_candidate_logits, dim=1)
        eq = torch.eq(predicted_idx, target_image)
        accuracy = float(eq.sum().data) / float(eq.nelement())

        return {
            'loss': communication_loss,
            'accuracy': 100 * accuracy,
            'message': message_dict['message_ids'],
            'mean_length': torch.mean(message_dict['message_lengths'].float())
        }


class ImageCaptionGrounder(CommunicationAgent):
    """
    An agent to train grounding of text to images, by producing captions based
    on images, and picking images based on a caption (independently)
    """
    def forward(self, batch):
        # Get the message dictionary (ids, logits, lengths) from the sender
        # based on the input image
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
            'attention_mask': batch['caption_mask']
        }
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
