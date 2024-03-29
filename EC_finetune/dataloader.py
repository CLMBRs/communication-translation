import random
from typing import List

from numpy import ndarray
from torch import LongTensor
from torch.utils.data.dataset import Dataset

from .util import *
from BackTranslation.constant import LANG_ID_2_LANGUAGE_CODES


class ImageIdentificationDataset(Dataset):
    """
    PyTorch Dataset subclass for image-identification games in which a "sender"
    agent takes in an image and communicates it, and a "receiver" identifies the
    correct image from among a selection of distractors
    Args:
        images: A numpy array of image data
        num_distractors: Number of distractor images to show to the "receiver"
            alongside the target image
    """
    def __init__(self, images: ndarray, num_distractors: int) -> Dataset:
        super().__init__()
        self.images = images
        self.img_index = list(range(len(images)))
        self.num_distractors = num_distractors

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> dict:
        # Randomly sample the distractor images out of the remainder of the
        # dataset
        distractor_candidates = (
            self.img_index[:index] + self.img_index[index + 1:]
        )
        distractor_images = (
            random.sample(distractor_candidates, k=self.num_distractors)
        )
        receiver_images = distractor_images + [index]
        random.shuffle(receiver_images)
        which = receiver_images.index(index)

        # Get the actual image embedding data to be returned
        sender_image = self.images[index]
        receiver_images = torch.index_select(
            self.images, 0, torch.tensor(receiver_images)
        )

        return {
            'sender_image': sender_image,
            'receiver_images': receiver_images,
            'target': which
        }


class XLImageIdentificationDataset(ImageIdentificationDataset):
    """
    PyTorch Dataset subclass for image-identification games in which a "sender"
    agent takes in an image and communicates it, and a "receiver" identifies the
    correct image from among a selection of distractors. This Class differs from
    the ImageIdentificationDataset by allowing user to condition the generation
    on target language (id) and vocabulary constraint.

    Args:
        images: A numpy array of image data
        num_distractors: Number of distractor images to show to the "receiver"
            alongside the target image
    """
    def __init__(
        self, images: ndarray, num_distractors: int, args, tokenizer
    ) -> Dataset:
        super().__init__(images, num_distractors)
        lang_code2id = dict(
            zip(
                tokenizer.additional_special_tokens,
                tokenizer.additional_special_tokens_ids
            )
        )
        self.source_lang_id = lang_code2id[args.language.source_lang]
        self.target_lang_id = lang_code2id[args.language.target_lang]
        self.lang_ids = [self.source_lang_id, self.target_lang_id]
        self.has_vocab_constraint = args.language.has_vocab_constraint

        if self.has_vocab_constraint:
            self.source_lang_mask = vocab_constraint_from_file(
                tokenizer, args.language.source_lang_vocab_constrain_file
            )
            self.target_lang_mask = vocab_constraint_from_file(
                tokenizer, args.language.target_lang_vocab_constrain_file
            )
            self.lang_masks = [self.source_lang_mask, self.target_lang_mask]

    def __getitem__(self, index: int) -> dict:
        # Randomly sample the distractor images out of the remainder of the
        # dataset
        ret = super().__getitem__(index)

        # choose a language to generate sentence for
        random_lang_idx = np.random.choice([0, 1])
        chosen_lang_id = self.lang_ids[random_lang_idx]
        ret['lang_id'] = chosen_lang_id
        if self.has_vocab_constraint:
            chosen_lang_mask = self.lang_masks[random_lang_idx]
            ret["lang_mask"] = chosen_lang_mask
        return ret


class CaptionTrainingDataset(ImageIdentificationDataset):
    """
    PyTorch Dataset subclass for training a model to generate a natural-language
    caption from an image embedding, and also pick the correct image from among
    distractors based on a caption
    Args:
        images: a numpy array of image data
        captions: a double-list of strings, where the first dimension
            correponds to the images, and the second dimension corresponds to
            the different captions for each image
        num_distractors: the number of distractor images to show alongside the
            target image
        tokenizer: the tokenizer for the model
        args: a namespace of additional arguments
    """
    def __init__(
        self,
        images: ndarray,
        captions: List[List[str]],
        num_distractors: int,
        tokenizer,
        args,
        max_length: int = 256,
        max_captions_per_image: int = float('inf')
    ) -> Dataset:
        # Initialize using the ImageIdentificationDataset constructor
        super().__init__(images, num_distractors)
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_length = max_length

        # The total number of training instances is the sum of caption options,
        # since an image can have more than one caption. Create a lookup
        # dictionary that returns a (image_index, caption_index) pair based on a
        # primary index. Also tokenize the captions
        self.caption_lookup = {}
        caption_index = 0

        for image_index, caption_set in enumerate(self.captions):
            for secondary_index in range(
                min(max_captions_per_image, len(caption_set))
            ):
                self.caption_lookup[caption_index] = (
                    image_index, secondary_index
                )
                caption_index += 1
        self.num_instances = len(self.caption_lookup)

        # Prepartion for language-constrained generation
        self.lang_code2id = dict(
            zip(
                tokenizer.additional_special_tokens,
                tokenizer.additional_special_tokens_ids
            )
        )
        self.lang_id = self.lang_code2id[args.language.source_lang]
        self.has_vocab_constraint = args.language.has_vocab_constraint

        if self.has_vocab_constraint:
            self.lang_mask = vocab_constraint_from_file(
                tokenizer, args.language.source_lang_vocab_constrain_file
            )

    def __len__(self) -> int:
        return self.num_instances

    def __getitem__(self, index: int) -> dict:
        # Get the image and caption-option index from the lookup table
        image_index, secondary_index = self.caption_lookup[index]
        # Use the supertype's __getitem__ to get the image, distractors, and
        # correct image index
        super_ret = super().__getitem__(image_index)
        caption = self.captions[image_index][secondary_index]
        caption = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        ret = {
            'sender_image': super_ret['sender_image'],
            'caption_ids': LongTensor(caption['input_ids']),
            'caption_mask': LongTensor(caption['attention_mask']),
            'receiver_images': super_ret['receiver_images'],
            'target': super_ret['target'],
            'lang_id': self.lang_id
        }
        if self.has_vocab_constraint:
            ret["lang_mask"] = self.lang_mask

        return ret


class TextInputECDataset(CaptionTrainingDataset):
    def __init__(
        self,
        images: ndarray,
        captions: List[List[str]],
        num_distractors: int,
        tokenizer,
        args,
        max_length: int = 256,
        max_captions_per_image: int = float('inf')
    ) -> Dataset:
        super().__init__(
            images, captions, num_distractors, tokenizer, args, max_length,
            max_captions_per_image
        )
        self.source_lang_id = self.lang_code2id[args.language.source_lang]
        self.target_lang_id = self.lang_code2id[args.language.target_lang]
        self.lang_ids = [self.source_lang_id, self.target_lang_id]

        if self.has_vocab_constraint:
            self.source_lang_mask = vocab_constraint_from_file(
                tokenizer, args.language.source_lang_vocab_constrain_file
            )
            self.target_lang_mask = vocab_constraint_from_file(
                tokenizer, args.language.target_lang_vocab_constrain_file
            )
            self.lang_masks = [self.source_lang_mask, self.target_lang_mask]

    def __getitem__(self, index: int) -> dict:
        # Randomly sample the distractor images out of the remainder of the
        # dataset
        super_ret = super().__getitem__(index)
        # choose a language to generate sentence for
        random_lang_idx = np.random.choice([0, 1])
        chosen_lang_id = self.lang_ids[random_lang_idx]
        ret = {
            'sender_input_text': super_ret['caption_ids'],
            'sender_attention_mask': super_ret['caption_mask'],
            'receiver_images': super_ret['receiver_images'],
            'target': super_ret['target'],
            'lang_id': chosen_lang_id
        }
        if self.has_vocab_constraint:
            chosen_lang_mask = self.lang_masks[random_lang_idx]
            ret['lang_mask'] = chosen_lang_mask
        return ret


class XLMDataset(Dataset):
    def __init__(
        self,
        examples: dict,
        tokenizer,
        alpha: float = 0.7,
        max_length: int = 128
    ) -> Dataset:
        super().__init__()
        self.examples = examples
        self.lang_ids = list(examples.keys())
        self.tokenizer = tokenizer
        self.alpha = alpha
        self.max_length = max_length

        # Use the original lengths of each dataset to calculate the resampling
        # probability of drawing from each language
        self.lengths = {
            lang_id: len(self.examples[lang_id])
            for lang_id in self.lang_ids
        }
        self.total_length = sum(self.lengths.values())

        self._set_sampling_lambdas()

        self.adjusted_lengths = {
            lang_id: round(self.lengths[lang_id] * self.lambdas[lang_id])
            for lang_id in self.lang_ids
        }
        self.adjusted_total_length = sum(self.adjusted_lengths.values())

        self._set_lang_indices()

    def __len__(self) -> int:
        return self.adjusted_total_length

    def __getitem__(self, index: int) -> dict:
        # Get the language for the batch based on the (up/down)-sampled language
        # set lengths
        sampled_lang = self._lang_from_idx(index)
        lang_fairseq_code = LANG_ID_2_LANGUAGE_CODES[sampled_lang]
        # The within-language index is set as the global index modulo the
        # language-specific length
        adjusted_index = index - self.lang_start_indices[sampled_lang]
        modulo_index = adjusted_index % self.lengths[sampled_lang]
        # Return the batch from the language-specific dataloader
        batch = self.examples[sampled_lang].__getitem__(modulo_index)
        self.tokenizer.set_src_lang_special_tokens(lang_fairseq_code)
        batch = self.tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask']
        }

    def _set_sampling_lambdas(self):
        orig_probs = {
            lang_id: (length / self.total_length)
            for lang_id, length in self.lengths.items()
        }
        exponiated_probs = {
            lang_id: prob**self.alpha
            for lang_id, prob in orig_probs.items()
        }
        z = sum(exponiated_probs.values())
        self.sample_probs = {
            lang_id: exponiated_probs[lang_id] / z
            for lang_id in self.lang_ids
        }
        self.lambdas = {
            lang_id: (1 / orig_probs[lang_id]) * self.sample_probs[lang_id]
            for lang_id in self.lang_ids
        }

    def _set_lang_indices(self):
        curr_start_index = 0
        self.lang_start_indices = {}
        for lang_id in self.lang_ids:
            self.lang_start_indices[lang_id] = curr_start_index
            curr_start_index = curr_start_index + self.adjusted_lengths[lang_id]

    def _lang_from_idx(self, index: int) -> str:
        for i, lang_id in enumerate(self.lang_ids):
            curr_start_index = self.lang_start_indices[lang_id]
            if i < len(self.lang_ids) - 1:
                next_lang_id = self.lang_ids[i + 1]
                next_start_index = self.lang_start_indices[next_lang_id]
            else:
                next_start_index = curr_start_index + self.adjusted_lengths[
                    lang_id]
            if curr_start_index <= index < next_start_index:
                return lang_id
        raise ValueError(f"Example index {index} is out of range")


class SingleLangXLMDataset(Dataset):
    def __init__(
        self, datafile: str, batch_size: int, order: str = 'none'
    ) -> Dataset:
        super().__init__()
        examples = [
            line.strip() for line in open(datafile, 'r') if line.strip() != ''
        ]
        if order == 'shuffle':
            random.shuffle(examples)
        elif order == 'sort':
            examples.sort(key=len, reverse=True)

        self.batch_size = batch_size
        num_examples = len(examples)
        self.num_batches = num_examples // self.batch_size
        num_examples = self.num_batches * self.batch_size
        examples = examples[:num_examples]

        self.batches = [
            examples[i:i + batch_size]
            for i in range(0, num_examples, batch_size)
        ]
        assert len(self.batches) == self.num_batches

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, index: int) -> List[list]:
        return self.batches[index]
