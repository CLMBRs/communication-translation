from abc import abstractmethod
from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module

from EC_finetune.speakers import Speaker
from EC_finetune.listeners import Listener


class CommunicationAgent(Module):
    """
    An abstract PyTorch Module for creating a sender/receiver or
    speaker/listener agent set for communication games

    Speaker and Listener are passed in as arguments, and must meet the
    interfaces specified by the Speaker and Listener abstract classes,
    respectively. Beholder (image embedding compoment) can either be shared
    between Speaker and Listener, or independent for both
    Args:
        speaker: the speaker/sender module, subclassed from Speaker
        listener: the listener/receiver module, subclassed from Listener
        args: a dictionary of parameters
    """
    def __init__(self, speaker: Speaker, listener: Listener, args: Namespace):
        super().__init__()

        # Initialize the image Beholder, and clone if there is to be a separate
        # Beholder stack for both Speaker and Listener
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

        self.speaker = speaker
        self.listener = listener

        self.image_dim = args.image_dim
        self.hidden_dim = args.hidden_dim
        self.unit_norm = args.unit_norm
        self.beam_width = args.beam_width
        self.no_share_bhd = args.no_share_bhd
        self.padding_index = args.padding_index

    def image_to_message(self, batch):
        # Embed the Speaker's image using the Beholder
        image_embedding = self.beholder1(batch['speaker_image'])

        # Generate the Speaker's message/caption about the image
        message_dict = self.speaker(image_embedding, **batch)
        return message_dict

    def choose_image_from_message(self, message_dict, listener_images):
        num_image_choices = listener_images.size(1)

        # Embed the Listener's candidate images using the Beholder
        listener_images = listener_images.view(-1, self.image_dim)
        listener_image_embeddings = self.beholder2(listener_images)
        listener_image_embeddings = listener_image_embeddings.view(
            -1, num_image_choices, self.hidden_dim
        )

        # Encode the Speaker's message to a hidden state to be used to select
        # a candidate image
        listener_hidden = self.listener(**message_dict)
        # (batch_size, num_image_choices, hidden_dim)
        listener_hidden = listener_hidden.unsqueeze(1).repeat(
            1, num_image_choices, 1
        )

        # Get the Mean Squared Error between the final listener representation
        # and all of the candidate images
        image_candidate_errors = F.mse_loss(
            listener_hidden, listener_image_embeddings, reduction='none'
        ).mean(dim=2).view(-1, num_image_choices)

        # Transform this to the inverted MSE (for use as scores)
        image_candidate_logits = 1 / (image_candidate_errors + 1e-10)
        return image_candidate_logits

    @abstractmethod
    def forward(self, batch):
        raise NotImplementedError

    
class ECImageIdentificationAgent(CommunicationAgent):
    def forward(self, batch):
        # Get the message dictionary (ids, logits, lengths) from the speaker
        # based on the input image
        message_dict = self.image_to_message(batch)

        # Get the logits for the image choice candidates based on the speaker's
        # message
        image_candidate_logits = self.choose_image_from_message(
            message_dict, batch['listener_images']
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
        # Get the message dictionary (ids, logits, lengths) from the speaker
        # based on the input image
        message_dict = self.image_to_message(batch)
        caption_generation_loss = F.cross_entropy(
            message_dict['message_logits'].transpose(1,2), batch['caption_ids'], ignore_index=self.padding_index
        )

        # Get the logits for the image choice candidates based on the gold
        # caption (NOT the speaker's message)
        caption = {'message_ids': batch['caption_ids']}
        image_candidate_logits = self.choose_image_from_message(
            caption, batch['listener_images']
        )
        # Get final cross-entropy loss between the candidates and the target
        # images
        target_image = batch['target']
        caption_understanding_loss = F.cross_entropy(
            image_candidate_logits, target_image
        )

        # Get predicted accuracy
        _, predicted_idx = torch.max(image_candidate_logits, dim=1)
        eq = torch.eq(predicted_idx, target_image)
        accuracy = float(eq.sum().data) / float(eq.nelement())

        loss = caption_generation_loss + caption_understanding_loss

        return {
            'loss': loss,
            'accuracy': 100 * accuracy,
            'message': message_dict['message_ids'],
            'mean_length': torch.mean(message_dict['message_lengths'].float())
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
