
from transformers import BartTokenizer, MBartTokenizer

from EC_finetune.src.speakers import *
from EC_finetune.src.listeners import *
from EC_finetune.src.modelings import BartForConditionalGeneration, MBartForConditionalGeneration


class ECAgent(torch.nn.Module):
    def __init__(self, args):
        super(ECAgent, self).__init__()
        if args.no_share_bhd:
            print("Not sharing visual system for each agent.")
            self.beholder1 = Beholder(args)
            self.beholder2 = Beholder(args)
        else:
            print("Sharing visual system for each agent.")
            self.beholder = Beholder(args)
        self.native, self.foreign = 'en', args.l2

        # Initialize speaker and listener
        if args.model == 'bart':
            tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
            model = BartForConditionalGeneration.from_pretrained(
                'facebook/bart-large'
            )

            self.speaker = BartSpeaker(model, self.native, args)
            self.listener = BartListener(model, self.foreign, args)

        elif args.model == 'rnn':
            self.speaker = RnnSpeaker(self.native, args)
            self.listener = RnnListener(self.foreign, args)
        elif args.model == 'mbart':
            tokenizer = MBartTokenizer.from_pretrained('facebook/mbart-large-cc25')
            model = MBartForConditionalGeneration.from_pretrained(
                'facebook/mbart-large-cc25'
            )

            self.speaker = MBartSpeaker(model, self.native, args)
            self.listener = MBartListener(model, self.foreign, args)

        self.tt = torch if args.cpu else torch.cuda
        self.native, self.foreign = 'en', args.l2
        self.unit_norm = args.unit_norm

        self.beam_width = args.beam_width
        self.norm_pow = args.norm_pow
        self.no_share_bhd = args.no_share_bhd
        self.D_img = args.D_img
        self.D_hid = args.D_hid

    def forward(self, batched_input, spk_sample_how):
        # spk_imgs : (batch_size, 2048)

        # speaker_image, listener_images, speaker_caps_in, speaker_cap_lens = batched_input
        speaker_image = batched_input["speaker_images"]
        listener_images = batched_input["listener_images"]
        speaker_caps_in = batched_input["speaker_caps_in"]
        speaker_cap_lens = batched_input["speaker_cap_lens"]

        num_dist = listener_images.size()[1]

        if self.no_share_bhd:
            speaker_images_hidden = self.beholder1(speaker_image)  # shared
        else:
            speaker_images_hidden = self.beholder(speaker_image)  # shared
        batched_input["speaker_images_hidden"] = speaker_images_hidden

        speaker_output = self.speaker(
            **batched_input
            # speaker_images_hidden, speaker_caps_in, speaker_cap_lens
        )  # NOTE argmax / gumbel
        speaker_message = speaker_output["generated_token_ids"]
        speaker_message_logits = speaker_output["generated_logits"]
        speaker_message_len = speaker_output["generated_sentence_len"]
        message_dict = {"speaker_message": speaker_message,
                        "speaker_message_logits": speaker_message_logits,
                        "speaker_message_len": speaker_message_len}

        lenlen = False
        if lenlen:
            print(speaker_message_len[:10])
            end_idx = torch.max(
                torch.ones(speaker_message_len.size()).cuda(),
                (speaker_message_len - 2).float()
            )
            end_idx_ = torch.arange(0, end_idx.size(0)
                                   ).cuda() * speaker_message_logits.size(1) + end_idx.int()

            end_loss_ = 3 * torch.ones(end_idx_.size()).long().cuda()
        else:
            end_idx_ = 0
            end_loss_ = 0

        listener_images = listener_images.view(-1, self.D_img)
        if self.no_share_bhd:
            listener_images_hidden = self.beholder2(listener_images)
        else:
            listener_images_hidden = self.beholder(listener_images)
        listener_images_hidden = listener_images_hidden.view(-1, num_dist, self.D_hid)

        listener_hiddens = self.listener(**message_dict)
        listener_hiddens = listener_hiddens.unsqueeze(1).repeat(
            1, num_dist, 1
        )  # (batch_size, num_dist, D_hid)

        # TODO: This is really bad style, need to fix when we figure out how
        return speaker_message_logits, (listener_hiddens,
                                        listener_images_hidden), speaker_message, (end_idx_, end_loss_), (
                                torch.min(speaker_message_len.float()),
                                torch.mean(speaker_message_len.float()),
                                torch.max(speaker_message_len.float())
                            )


class Beholder(torch.nn.Module):
    def __init__(self, args):
        super(Beholder, self).__init__()
        self.image_to_hidden = torch.nn.Linear(args.D_img, args.D_hid)
        self.unit_norm = args.unit_norm
        self.dropout = nn.Dropout(p=args.dropout)
        self.two_ffwd = args.two_ffwd
        if self.two_ffwd:
            self.hidden_to_hidden = torch.nn.Linear(args.D_hid, args.D_hid)

    def forward(self, image):
        h_image = image
        h_image = self.image_to_hidden(h_image)
        h_image = self.dropout(h_image)

        if self.two_ffwd:
            h_image = self.hidden_to_hidden(F.relu(h_image))

        if self.unit_norm:
            norm = torch.norm(h_image, p=2, dim=1, keepdim=True).detach() + 1e-9
            h_image = h_image / norm.expand_as(h_image)
        return h_image
