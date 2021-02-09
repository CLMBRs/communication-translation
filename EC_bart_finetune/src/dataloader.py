import random
import torch
import numpy as np
from numpy import ndarray
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from util import *


class ImageIdentificationDataset(Dataset):
    """
    PyTorch Dataset subclass for image-identification games in which a "speaker"
    agent takes in an image and communicates it, and a "listener" identifies the
    correct image from among a selection of distractors

    Args:
        images: A NumPy array of image data??
        num_distractors: Number of distractor images to show to the "listener"
            alongside the target image
    """
    def __init__(self, images: ndarray, num_distractors: int) -> Dataset:
        super(MyDataset, self).__init__()
        self.images = images
        self.img_index = list(range(len(images)))
        self.num_distractors = num_distractors

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        # Randomly sample the distractor images out of the remainder of the
        # dataset
        distractor_candidates = (
            self.img_index[:index] + self.img_index[index + 1:]
        )
        distractor_imgages = (
            random.sample(dist_candidates, k=self.num_distractors)
        )
        listener_images = distractor_images + [index]
        random.shuffle(listener_images)
        which = listener_images.index(index)

        # Get the actual image embedding data to be returned
        speaker_image = self.images[index]
        listener_images = torch.index_select(
            self.images, 0, torch.tensor(listener_images)
        ).numpy()

        # TODO: what are all these zeros?
        return (speaker_image, listener_images, 0, 0, 0, 0, 0, which)


def next_batch_joint(images, batch_size, num_distractors):
    # TODO: is this entire function deprecated in favor of our data loader??
    speaker_images, speaker_caps, listener_images, listener_caps, whichs = (
        [], [], [], [], []
    )
    assert len(images) - 1 >= num_distractors
    for batch_idx in range(batch_size):
        image_indices = random.permutation(len(images))[:num_dist]
        # Shape: (1)
        which = random.randint(0, num_dist)
        speaker_image = image_indices[which]
        # Shape: (batch_size, 2048)
        speaker_images.append(speaker_image)
        # Shape: batch_size * num_dist
        listener_images += list(image_indices)
        # Shape: (batch_size)
        whichs.append(which)

    speaker_images = torch.index_select(
        images, 0, torch.tensor(speaker_images)
    ).view(batch_size, -1)
    listener_images = torch.index_select(
        images, 0, torch.tensor(listener_images)
    ).view(batch_size, num_dist, -1)
    whichs = torch.LongTensor(whichs).view(batch_size, -1)
    
    return (spk_imgs, lsn_imgs, 0, 0, 0, 0, 0, whichs)


def weave_out(caps_out):
    # TODO: figure out why this function exists
    ans = []
    seq_len = max([len(x) for x in caps_out])
    for idx in range(seq_len):
        for sublst in caps_out:
            if idx < len(sublst):
                ans.append(sublst[idx])
    return ans
