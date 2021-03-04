from argparse import Namespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Module

from EC_finetune.speakers import BartSpeaker, MBartSpeaker, RnnSpeaker
from EC_finetune.listeners import Listener, BartEncoder, MBartEncoder, RnnEncoder
from EC_finetune.modelings.modeling_bart import BartForConditionalGeneration
from EC_finetune.modelings.modeling_mbart import MBartForConditionalGeneration


class CommunicationAgent(Module):
    """
    A PyTorch module instantiating a sender/receiver or speaker/listener agent
    set for communication games
    Speaker and Listener components can currently be set to have either an RNN
    or pretrained Bart architecture. Beholder (image embedding compoment) can
    either be shared between Speaker and Listener, or independent for both
    Args:
        args: A dictionary of parameters
    """
    def __init__(self, args: Namespace):
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
        self.native, self.foreign = 'en', args.l2
        if args.no_share_bhd:
            print("Not sharing visual system for each agent.")
            self.beholder1 = self.beholder
            self.beholder2 = self.beholder1.clone()
        else:
            print("Sharing visual system for each agent.")
            self.beholder1 = self.beholder2 = self.beholder

            # Initialize Speaker and Listener, either from pretrained Bart or as a
        # from-scratch RNN
        # TODO: Have RNN stack be shared between Speaker and Listener
        if args.model_name == 'bart':
            self.model = BartForConditionalGeneration.from_pretrained(
                'facebook/bart-large'
            )
            self.speaker = BartSpeaker(self.model, self.native, **vars(args))
            listener_model = BartEncoder(self.model, args.hidden_dim)
        elif args.model_name == "mbart":
            self.model = MBartForConditionalGeneration.from_pretrained(
                'facebook/mbart-large-cc25'
            )
            self.speaker = MBartSpeaker(self.model, self.native, **vars(args))
            listener_model = MBartEncoder(self.model, args.hidden_dim)
        elif args.model_name == 'rnn':
            self.speaker = RnnSpeaker(self.native, args)
            listener_model = RnnEncoder(
                args.vocab_size, args.embedding_dim, args.hidden_dim,
                args.num_layers, args.bidirectional
            )
            self.model = None
        else:
            raise ValueError(f"Model type {args.model_name} is not valid")

        self.listener = Listener(
            listener_model, dropout=args.dropout, unit_norm=args.unit_norm
        )

        self.crossentropy = torch.nn.CrossEntropyLoss()

        self.image_dim = args.image_dim
        self.hidden_dim = args.hidden_dim
        self.unit_norm = args.unit_norm
        self.beam_width = args.beam_width
        self.norm_pow = args.norm_pow
        self.no_share_bhd = args.no_share_bhd

    def forward(self, batch):
        target_image = batch['target']
        speaker_images = batch['speaker_image']
        listener_images = batch['listener_images']
        speaker_captions_in = batch['speaker_caps_in']
        speaker_caption_lengths = batch['speaker_cap_lens']

        num_image_choices = listener_images.size(1)

        # Embed the Speaker's image using the Beholder
        speaker_image_embeddings = self.beholder1(speaker_images)
        batch["speaker_images_hidden"] = speaker_image_embeddings
        # input_dict = {"speaker_images_hidden": speaker_image_embeddings}
        # if "lang_ids" in batch:
        #     input_dict["lang_ids"] = batch["lang_ids"]
        # if "lang_masks" in batch:
        #     input_dict["lang_masks"] = batch["lang_masks"]

        # Generate the Speaker's message/caption about the image. (The speaker
        # classes are currently not accepting input captions)
        message_dict = self.speaker(**batch)
        message_ids = message_dict["message_ids"]
        message_logits = message_dict["message_logits"]
        message_lengths = message_dict["message_lengths"]
        '''
        message_logits, message_ids, message_lengths = self.speaker(
            speaker_image_embeddings, speaker_captions_in, speaker_caption_lengths
        ) 
        '''

        # Commenting this out until we know why it's here
        '''
        lenlen = False
        if lenlen:
            print(spk_cap_len_[:10])
            end_idx = torch.max(
                torch.ones(spk_cap_len_.size()).cuda(),
                (spk_cap_len_ - 2).float()
            )
            end_idx_ = torch.arange(0, end_idx.size(0)
                                   ).cuda() * spk_logits.size(1) + end_idx.int()
            end_loss_ = 3 * torch.ones(end_idx_.size()).long().cuda()
        else:
        '''
        end_idx_ = 0
        end_loss_ = 0

        # Embed the Listener's candidate images using the Beholder
        listener_images = listener_images.view(-1, self.image_dim)
        listener_image_embeddings = self.beholder2(listener_images)
        # TODO: this is a problem
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

        # Get final cross-entropy loss
        communication_loss = self.crossentropy(
            image_candidate_logits, target_image
        )
        # Get predicted accuracy
        _, predicted_idx = torch.max(image_candidate_logits, dim=1)
        eq = torch.eq(predicted_idx, target_image)
        accuracy = float(eq.sum().data) / float(eq.nelement())

        return_dict = {
            'loss': communication_loss,
            'accuracy': accuracy,
            'message': message_ids,
            'end_idx': end_idx_,
            'end_loss': end_loss_,
            'mean_length': torch.mean(message_lengths.float())
        }

        return return_dict
        # return speaker_message_logits, (listener_hiddens,
        #                                 listener_images_hidden), speaker_message, (end_idx_, end_loss_), (
        #                         torch.min(speaker_message_len.float()),
        #
        #                         torch.max(speaker_message_len.float())
        #                     )


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
