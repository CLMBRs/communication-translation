import torch.nn as nn
from torch.nn import Module


class BartSpeaker(Module):
    def __init__(
        self,
        bart,
        lang,
        hidden_dim=None,
        embedding_dim=None,
        seq_len=None,
        temperature=None,
        hard=None,
        **kwargs
    ):
        super().__init__()
        self.lang = lang
        self.speaker = model
        self.projection = nn.Linear(hidden_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.speaker.temp = temperature
        self.speaker.hard = hard
        self.seq_len = seq_len

    def forward(self, speaker_images_hidden, **kwargs):
        batch_size = speaker_images_hidden.size()[0]
        assert len(speaker_images_hidden.shape) == 2   # (batch_size, image_hidden_dim)
        speaker_images_hidden = speaker_images_hidden.unsqueeze(1).repeat(1, self.seq_len, 1)
        speaker_images_hidden = self.projection(speaker_images_hidden)
        # h_image = h_image.view(-1, self.seq_len, self.embedding_dim)
        if "lang_id" in kwargs:
            kwargs["lang_id"] = kwargs["lang_id"].view(batch_size, -1)
        output = self.speaker.gumbel_generate(
            input_images=speaker_images_hidden,
            num_beams=1,
            max_length=self.seq_len,
            **kwargs
        )
        return {
            "message_ids": output["generated_token_ids"],
            "message_logits": output["generated_logits"],
            "message_lengths": output["generated_sentence_len"],
            # "message_embeddings": output["generated_embeddings"]
        }


MBartSpeaker = BartSpeaker
