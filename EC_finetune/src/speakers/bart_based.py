import torch
import torch.nn as nn


class BartSpeaker(torch.nn.Module):
    def __init__(self, bart, lang, args):
        super(BartSpeaker, self).__init__()
        self.spk = bart
        self.project = nn.Linear(args.D_hid, args.seq_len * args.D_emb)
        self.D_emb = args.D_emb

        self.spk.temp = args.temp
        self.spk.hard = args.hard
        self.spk.tt = torch if args.cpu else torch.cuda
        self.seq_len = args.seq_len

    def forward(self, speaker_images_hidden, **kwargs):
        # caps_in.size()[0]
        batch_size = speaker_images_hidden.size()[0]

        speaker_images_hidden = self.project(speaker_images_hidden)
        speaker_images_hidden = speaker_images_hidden.view(-1, self.seq_len, self.D_emb)
        output = self.spk.gumbel_generate(
            input_images=speaker_images_hidden, num_beams=1, max_length=self.seq_len
        )
        return {
            "speaker_message": output["generated_token_ids"],
            "speaker_message_logits": output["generated_logits"],
            "speaker_message_len": output["generated_sentence_len"]
        }


class MBartSpeaker(torch.nn.Module):
    def __init__(self, mbart, lang, args):
        super(MBartSpeaker, self).__init__()
        self.spk = mbart
        self.project = nn.Linear(args.D_hid, args.seq_len * args.D_emb)

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

    def forward(self, speaker_images_hidden, lang_ids, lang_masks, **kwargs):
        # here we assume lang_ids and masks are also batched and given
        # TODO: how to pack the input for a unified framework?

        batch_size = speaker_images_hidden.size()[0]  # caps_in.size()[0]
        assert batch_size == len(lang_ids) == len(lang_masks)

        speaker_images_hidden = self.project(speaker_images_hidden)
        speaker_images_hidden = speaker_images_hidden.view(-1, self.seq_len, self.D_emb)
        output = self.spk.gumbel_generate(input_images=speaker_images_hidden,
                                          num_beams=1,
                                          max_length=self.seq_len,
                                          lang_masks=lang_masks,
                                          lang_ids=lang_ids.view(batch_size, -1))

        return {
            "speaker_message": output["generated_token_ids"],
            "speaker_message_logits": output["generated_logits"],
            "speaker_message_len": output["generated_sentence_len"]
        }
