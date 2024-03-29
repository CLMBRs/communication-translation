import os
import torch
import random
from numpy import ndarray
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import MBartTokenizer
from EC_finetune.util import vocab_constraint_from_file
from BackTranslation.constant import FAIRSEQ_LANGUAGE_CODES
from ipdb import set_trace as bp


class MbartMonolingualDataset(Dataset):
    """
    PyTorch Dataset subclass for image-identification games in which a "speaker"
    agent takes in an image and communicates it, and a "listener" identifies the
    correct image from among a selection of distractors
    Args:
        images: A NumPy array of image data
        num_distractors: Number of distractor images to show to the "listener"
            alongside the target image
    """
    def __init__(self, source_file: str,
                 tokenizer: MBartTokenizer,
                 lang_code: str,):
        super(MbartMonolingualDataset, self).__init__()

        self.dataset = []
        # bp()
        assert os.path.exists(source_file)
        self.source_file = source_file
        assert lang_code in FAIRSEQ_LANGUAGE_CODES
        self.lang_code = lang_code
        tokenizer.set_src_lang_special_tokens(lang_code)
        self.tokenizer = tokenizer

        with open(source_file, "r") as f:
            for line in tqdm(f):
                line = line.strip()
                if line == "":
                    continue
                # encoded_sentence = torch.tensor(self.tokenizer.encode(line))  # tokenized + indexed
                self.dataset.append(line)
        random.shuffle(self.dataset)


    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        return self.dataset[index]
