import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from transformers import BartTokenizer

from gumbel_utils import gumbel_softmax
from modeling_bart import BartForConditionalGeneration
from util import *


class CommunicationAgent(torch.nn.Module):
    def __init__(self, args):
        super(CommunicationAgent, self).__init__()

        self.beholder = Beholder(
            image_dim=args.D_img,
            hidden_dim=args.D_hid,
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

        # Initialize speaker and listener
        if args.model == 'bart':
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            model = BartForConditionalGeneration.from_pretrained(
                'facebook/bart-large'
            )
            self.speaker = BartSpeaker(model, self.native, args)
            self.listener = BartListener(
                hidden_dim=args.D_hid,
                embedding_dim=args.D_emb,
                dropout=args.dropout,
                bart=model,
                unit_norm=args.unit_norm
            )
        elif args.model == 'rnn':
            self.speaker = RnnSpeaker(self.native, args)
            self.listener = RnnListener(
                hidden_dim=args.D_hid,
                embedding_dim=args.D_emb,
                dropout=args.dropout,
                num_layers=args.num_layers,
                vocab_size=args.vocab_size,
                unit_norm=args.unit_norm,
                bidirectional=args.bidirectional
            )
        else:
            raise ValueError(f"Model type {args.model} is not valid")

        self.native, self.foreign = 'en', args.l2

        self.image_dim = args.D_img
        self.hidden_dim = args.D_hid
        self.unit_norm = args.unit_norm
        self.beam_width = args.beam_width
        self.norm_pow = args.norm_pow
        self.no_share_bhd = args.no_share_bhd

    def forward(self, data1, spk_sample_how):
        # spk_imgs : (batch_size, 2048)
        a_spk_img, b_lsn_imgs, a_spk_caps_in, a_spk_cap_lens = data1

        num_dist = b_lsn_imgs.size()[1]

        if self.no_share_bhd:
            spk_h_img = self.beholder1(a_spk_img)  # not shared
        else:
            spk_h_img = self.beholder(a_spk_img)  # shared

        spk_logits, spk_msg, spk_cap_len_ = self.speaker(
            spk_h_img, a_spk_caps_in, a_spk_cap_lens
        )  # NOTE argmax / gumbel

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

        lsn_imgs = b_lsn_imgs.view(-1, self.image_dim)
        if self.no_share_bhd:
            lsn_h_imgs = self.beholder2(lsn_imgs)
        else:
            lsn_h_imgs = self.beholder(lsn_imgs)
        lsn_h_imgs = lsn_h_imgs.view(-1, num_dist, self.hidden_dim)
        lis_hid = self.listener(spk_msg, spk_cap_len_, spk_logits)
        lis_hid = lis_hid.unsqueeze(1).repeat(
            1, num_dist, 1
        )  # (batch_size, num_dist, D_hid)

        # TODO: This is really bad style, need to fix when we figure out how
        return spk_logits, (lis_hid,
                            lsn_h_imgs), spk_msg, (end_idx_, end_loss_), (
                                torch.min(spk_cap_len_.float()),
                                torch.mean(spk_cap_len_.float()),
                                torch.max(spk_cap_len_.float())
                            )


class Beholder(torch.nn.Module):
    """
    A Beholder module for embedding image data. Consists of one or two
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
        super(Beholder, self).__init__()
        self.image_to_hidden = torch.nn.Linear(image_dim, hidden_dim)
        self.unit_norm = unit_norm
        self.dropout = nn.Dropout(p=dropout)
        self.two_ffwd = two_ffwd
        if self.two_ffwd:
            self.hidden_to_hidden = torch.nn.Linear(hidden_dim, hidden_dim)

    def forward(self, image: Tensor) -> Tensor:
        """
        Return the forward output of the Beholder nn module

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


class Listener(torch.nn.Module):
    def __init__(self, hidden_dim, embedding_dim, dropout, unit_norm=False):
        super(Listener, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.unit_norm = unit_norm
        self.dropout = nn.Dropout(p=dropout)
        self.listener = None
        self.embedding = None


class BartListener(Listener):
    """
    A "listener" module based on a pretrained BART model
    """
    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        bart,
        unit_norm: bool = False
    ) -> Listener:
        super(BartListener, self).__init__(
            hidden_dim, embedding_dim, dropout, unit_norm=unit_norm
        )
        self.hidden_to_hidden = nn.Linear(1024, hidden_dim)
        self.listener = bart.gumbel_encoder
        self.embedding = bart.model.shared

    def forward(self, spk_msg, spk_logit=0):
        # spk_msg : (batch_size, seq_len)
        # spk_msg_lens : (batch_size)
        spk_msg_emb = torch.matmul(spk_logit, self.embedding.weight)
        spk_msg_emb = self.dropout(spk_msg_emb)

        output = self.listener(spk_msg, spk_msg_emb)
        # Mean pooling for now to match the img output
        output = torch.mean(output.last_hidden_state, dim=1)

        # Transform the dim to match the img dim
        out = self.hidden_to_hidden(output)

        return out


class RnnListener(Listener):
    def __init__(
        self,
        hidden_dim: int,
        embedding_dim: int,
        dropout: float,
        num_layers: int,
        vocab_size: int,
        unit_norm: bool = False,
        bidirectional: bool = False
    ) -> Listener:
        super(RnnListener, self).__init__(
            hidden_dim, embedding_dim, dropout, unit_norm=unit_norm
        )
        self.listener = nn.GRU(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        num_directions = 2 if bidirectional else 1
        self.hid_to_hid = nn.Linear(num_directions * hidden_dim, hidden_dim)

        self.num_layers = num_layers
        self.num_directions = num_directions
        self.vocab_size = vocab_size

    def forward(self, spk_msg, spk_msg_lens, spk_logit=0):
        # spk_msg : (batch_size, seq_len)
        # spk_msg_lens : (batch_size)
        batch_size = spk_msg.size()[0]

        h_0 = Variable(
            torch.zeros(
                self.num_layers * self.num_directions, batch_size,
                self.hidden_dim
            )
        )

        spk_msg_emb = torch.matmul(spk_logit, self.embedding.weight)
        spk_msg_emb = self.drop(spk_msg_emb)

        pack = torch.nn.utils.rnn.pack_padded_sequence(
            spk_msg_emb,
            spk_msg_lens.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, h_n = self.listener(pack, h_0)

        h_n = h_n[-self.num_directions:, :, :]
        out = h_n.transpose(0, 1).contiguous().view(
            batch_size, self.num_directions * self.hidden_dim
        )
        out = self.hid_to_hid(out)

        if self.unit_norm:
            norm = torch.norm(out, p=2, dim=1, keepdim=True).detach() + 1e-9
            out = out / norm.expand_as(out)

        return out


class BartSpeaker(torch.nn.Module):
    def __init__(self, bart, embedding_dim, seq_len, temperature, hard):
        super(BartSpeaker, self).__init__()
        self.spk = bart
        self.projection = nn.Linear(embedding_dim, seq_len * embedding_dim)
        self.embedding_dim = embedding_dim

        self.spk.temp = temperature
        self.spk.hard = hard
        self.seq_len = seq_len

    def forward(self, h_image, caps_in, caps_in_lens):
        h_image = self.projection(h_image)
        h_image = h_image.view(-1, self.seq_len, self.embedding_dim)
        input_ids, input_logits, cap_len = self.spk.gumbel_generate(
            input_images=h_image, num_beams=1, max_length=self.seq_len
        )
        return input_logits, input_ids, cap_len


class RnnSpeaker(torch.nn.Module):
    def __init__(self, lang, args):
        super(RnnSpeaker, self).__init__()
        self.rnn = nn.GRU(
            args.D_emb, args.D_hid, args.num_layers, batch_first=True
        )
        self.emb = nn.Embedding(args.vocab_size, args.D_emb, padding_idx=0)

        self.hid_to_voc = nn.Linear(args.D_hid, args.vocab_size)

        self.D_emb = args.D_emb
        self.D_hid = args.D_hid
        self.num_layers = args.num_layers
        self.drop = nn.Dropout(p=args.dropout)

        self.vocab_size = args.vocab_size
        self.temp = args.temp
        self.hard = args.hard
        self.tt = torch if args.cpu else torch.cuda
        self.tt_ = torch
        self.seq_len = args.seq_len

    def forward(self, h_img, caps_in, caps_in_lens):

        batch_size = h_img.size()[0]  # caps_in.size()[0]

        h_img = h_img.view(1, batch_size,
                           self.D_hid).repeat(self.num_layers, 1, 1)

        initial_input = self.emb(
            torch.ones([batch_size, 1], dtype=torch.int64).cuda() * 2
        )
        out_, hid_ = self.rnn(initial_input, h_img)
        logits_ = []
        labels_ = []
        for idx in range(self.seq_len):
            logit_ = self.hid_to_voc(out_.view(-1, self.D_hid))
            c_logit_, comm_label_ = gumbel_softmax(
                logit_, self.temp, self.hard, self.tt, idx
            )

            input_ = torch.matmul(c_logit_.unsqueeze(1), self.emb.weight)
            out_, hid_ = self.rnn(input_, hid_)
            logits_.append(c_logit_.unsqueeze(1))
            labels_.append(comm_label_)
        logits_ = torch.cat(logits_, dim=1)
        labels_ = torch.cat(labels_, dim=-1)
        tmp = torch.zeros(logits_.size(-1))
        tmp[3] = 1
        logits_[:, -1, :] = tmp
        labels_[:, -1] = 3
        pad_g = ((labels_ == 3).cumsum(1) == 0)
        labels_ = pad_g * labels_
        pad_ = torch.zeros(logits_.size()).cuda()
        pad_[:, :, 0] = 1
        logits_ = torch.where(
            pad_g.unsqueeze(-1).repeat(1, 1, logits_.size(-1)), logits_, pad_
        )

        cap_len = pad_g.cumsum(1).max(1).values + 1

        return logits_, labels_, cap_len
