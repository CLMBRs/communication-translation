import random
from typing import List

from numpy import ndarray
from torch.utils.data.dataset import Dataset

from EC_finetune.util import *


class ImageIdentificationDataset(Dataset):
    """
    PyTorch Dataset subclass for image-identification games in which a "speaker"
    agent takes in an image and communicates it, and a "listener" identifies the
    correct image from among a selection of distractors
    Args:
        images: A numpy array of image data
        num_distractors: Number of distractor images to show to the "listener"
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
        listener_images = distractor_images + [index]
        random.shuffle(listener_images)
        which = listener_images.index(index)

        # Get the actual image embedding data to be returned
        speaker_image = self.images[index]
        listener_images = torch.index_select(
            self.images, 0, torch.tensor(listener_images)
        )

        return {
            'speaker_image': speaker_image,
            'listener_images': listener_images,
            'target': which
        }


class VisuaLingConstraintDataset(ImageIdentificationDataset):
    """
    PyTorch Dataset subclass for image-identification games in which a "speaker"
    agent takes in an image and communicates it, and a "listener" identifies the
    correct image from among a selection of distractors. This Class differs from
    the ImageIdentificationDataset by allowing user to condition the generation
    on target language (id) and vocabulary constraint.

    Args:
        images: A numpy array of image data
        num_distractors: Number of distractor images to show to the "listener"
            alongside the target image
    """
    def __init__(self, images: ndarray, num_distractors: int, args, tokenizer):
        super().__init__()
        self.images = images
        self.img_index = list(range(len(images)))
        self.num_distractors = num_distractors
        lang_code2id = dict(
            zip(
                tokenizer.additional_special_tokens,
                tokenizer.additional_special_tokens_ids
            )
        )
        self.source_lang_id = lang_code2id[args.source_lang]
        self.target_lang_id = lang_code2id[args.target_lang]
        self.lang_ids = [self.source_lang_id, self.target_lang_id]
        self.has_vocab_constraint = args.has_vocab_constraint

        if self.has_vocab_constraint:
            self.source_lang_mask = vocab_mask_from_file(
                tokenizer, args.source_lang_vocab_constrain_file
            )
            self.target_lang_mask = vocab_mask_from_file(
                tokenizer, args.target_lang_vocab_constrain_file
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
        self, images: ndarray, captions: List[List[str]], num_distractors: int,
        tokenizer, args
    ) -> Dataset:
        # Initialize using the ImageIdentificationDataset constructor
        super().__init__(images, num_distractors)
        self.captions = captions
        
        # The total number of training instances is the sum of caption options,
        # since an image can have more than one caption. Create a lookup
        # dictionary that returns a (image_index, caption_index) pair based on a
        # primary index. Also tokenize the captions
        self.num_instances = sum([len(options) for options in self.captions])
        self.caption_lookup = {}
        caption_index = 0
        for image_index, caption_set in enumerate(self.captions):
            for secondary_index in range(len(caption_set)):
                self.caption_lookup[caption_index] = (
                    image_index, secondary_index
                )
                caption_dict = tokenizer(
                    [self.captions[image_index][secondary_index]]
                )
                caption_dict['message_ids'] = caption_dict['input_ids']
                self.captions[image_index][secondary_index] = caption_dict
                caption_index += 1

        # Prepartion for language-constrained generation
        lang_code2id = dict(
            zip(
                tokenizer.additional_special_tokens,
                tokenizer.additional_special_tokens_ids
            )
        )
        self.lang_id = lang_code2id[args.source_lang]
        self.has_vocab_constraint = args.has_vocab_constraint
        if self.has_vocab_constraint:
            self.lang_mask = vocab_mask_from_file(
                tokenizer, args.source_lang_vocab_constrain_file
            )

    def __len__(self) -> int:
        return self.num_instances

    def __getitem__(self, index: int) -> dict:
        # Get the image and caption-option index from the lookup table
        image_index, secondary_index = self.caption_lookup[index]
        # Use the supertype's __getitem__ to get the image, distractors, and 
        # correct image index
        super_ret = super().__getitem__(image_index)
        ret = {
            'speaker_image': super_ret['speaker_image'],
            'caption': self.captions[image_index][secondary_index],
            'listener_images': super_ret['listener_images'],
            'target': super_ret['target'],
            'lang_id': self.lang_id
        }
        if self.has_vocab_constraint:
            ret["lang_mask"] = self.lang_mask

        return ret
