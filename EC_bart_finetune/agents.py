import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn import Module

from modeling_bart import BartForConditionalGeneration
from util import *


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
    def __init__(self, args: dict):
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

        # Initialize Speaker and Listener, either from pretrained Bart or as a
        # from-scratch RNN
        # TODO: Have RNN stack be shared between Speaker and Listener
        if args.model == 'bart':
            self.model = BartForConditionalGeneration.from_pretrained(
                'facebook/bart-large'
            )
            self.speaker = BartSpeaker(self.model, self.native, args)
            listener_model = BartEncoder(self.model, args.hidden_dim)
        elif args.model == 'rnn':
            self.speaker = RnnSpeaker(self.native, args)
            listener_model = RnnEncoder(
                args.vocab_size, args.embedding_dim, args.hidden_dim,
                args.num_layers, args.bidirectional
            )
            self.model = None
        else:
            raise ValueError(f"Model type {args.model} is not valid")

        self.listener = Listener(
            listener_model, dropout=args.dropout, unit_norm=args.unit_norm
        )

        self.crossentropy = torch.nn.CrossEntropyLoss()

        self.native, self.foreign = 'en', args.l2

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
        speaker_image_embeddings = (
            self.beholder1(speaker_images) if self.no_share_bhd
            else self.beholder(speaker_images)
        )

        # Generate the Speaker's message/caption about the image. (The speaker
        # classes are currently not accepting input captions)
        message_logits, message_ids, message_lengths = self.speaker(
            speaker_image_embeddings
        )
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
        listener_image_embeddings = (
            self.beholder2(speaker_images) if self.no_share_bhd
            else self.beholder(speaker_images)
        )
        listener_image_embeddings = listener_image_embeddings.view(
            -1, num_image_choices, self.hidden_dim
        )

        # Encode the Speaker's message to a hidden state to be used to select
        # a candidate image
        listener_hidden = self.listener(message_logits, message_lengths)
        # (batch_size, num_image_choices, hidden_dim)
        listener_hidden = listener_hidden.unsqueeze(1).repeat(
            1, num_image_choices, 1
        )

        # Get the Mean Squared Error between the final listener representation
        # and all of the candidate images
        image_candidate_errors = F.mse_loss(
            listener_hidden, listener_images, reduction='none'
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
        self, message_logits: Tensor, message_lengths: Tensor
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
        output = self.listener(message_embedding, message_lengths)

        if self.unit_norm:
            norm = torch.norm(output, p=2, dim=1, keepdim=True).detach() + 1e-9
            output = output / norm.expand_as(output)

        return output


# TODO: Speaker can also probably be collapsed into a single class that defines
# the common forward components. Then the passable model would be one that takes
# in an image and returns a generated message/caption


class BartSpeaker(Module):
    def __init__(self, bart, embedding_dim, seq_len, temperature, hard):
        super().__init__()
        self.speaker = bart
        self.projection = nn.Linear(embedding_dim, seq_len * embedding_dim)
        self.embedding_dim = embedding_dim

        self.speaker.temp = temperature
        self.speaker.hard = hard
        self.seq_len = seq_len

    def forward(self, h_image):
        h_image = self.projection(h_image)
        h_image = h_image.view(-1, self.seq_len, self.embedding_dim)
        message_ids, message_logits, message_lengths = self.speaker.gumbel_generate(
            input_images=h_image, num_beams=1, max_length=self.seq_len
        )
        return message_logits, message_ids, message_lengths


class RnnSpeaker(Module):
    def __init__(self, args):
        super().__init__()
        self.rnn = nn.GRU(
            args.embedding_dim,
            args.hidden_dim,
            args.num_layers,
            batch_first=True
        )
        self.embedding = nn.Embedding(
            args.vocab_size, args.embedding_dim, padding_idx=0
        )
        self.hiden_to_vocab = nn.Linear(args.hidden_dim, args.vocab_size)
        self.dropout = nn.Dropout(p=args.dropout)

        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.num_layers = args.num_layers
        self.vocab_size = args.vocab_size
        self.bos_idx = args.bos_idx
        self.temp = args.temp
        self.hard = args.hard
        self.seq_len = args.seq_len

    def forward(self, h_image):
        batch_size = h_image.size(0)  # caps_in.size()[0]

        # Embed the image representation for use as the initial hidden state
        # for all layers of the RNN
        h_image = h_image.view(1, batch_size,
                               self.hidden_dim).repeat(self.num_layers, 1, 1)

        # Set the initial input for generation to be the BOS index, and run this
        # through the RNN to get the first output and hidden state
        initial_input = self.embedding(
            torch.ones([batch_size, 1]).cuda() * self.bos_idx
        )
        output, hidden = self.rnn(initial_input, h_image)

        # Loop through generation until the message is length seq_len. At each
        # step, get the logits over the vocab size, pass this through Gumbel
        # Softmax to get the generation distribution and the predicted vocab
        # id. Pass the predicted id to generate the next step
        logits = []
        labels = []
        for idx in range(self.seq_len):
            logit = self.hidden_to_vocab(output.view(-1, self.hidden_dim))
            logit_sample, label = F.gumbel_softmax(
                logit, tau=self.temp, hard=self.hard
            )
            next_input = torch.matmul(
                logit_sample.unsqueeze(1), self.embedding.weight
            )
            output, hidden = self.rnn(next_input, hidden)
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

        return message_logits, message_ids, message_lengths


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
        self, message_embedding: Tensor, message_lengths: Tensor
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
        hidden = self.encoder(message_embedding)
        pooled_hidden = torch.mean(hidden.last_hidden_state, dim=1)
        output = self.hidden_to_output(pooled_hidden)
        return output


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

        h_0 = Variable(
            torch.zeros(
                self.num_layers * self.num_directions, batch_size,
                self.output_dim
            )
        )

        pack = torch.nn.utils.rnn.pack_padded_sequence(
            message_embedding,
            message_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, representation = self.rnn(pack, h_0)

        representation = representation[-self.num_directions:, :, :]
        output = representation.transpose(0, 1).contiguous().view(
            batch_size, self.num_directions * self.output_dim
        )
        output = self.hidden_to_output(output)
        return output
