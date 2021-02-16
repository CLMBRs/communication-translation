import random
import torch
from numpy import ndarray
from torch.utils.data.dataset import Dataset


class ImageIdentificationDataset(Dataset):
    """
    PyTorch Dataset subclass for image-identification games in which a "speaker"
    agent takes in an image and communicates it, and a "listener" identifies the
    correct image from among a selection of distractors

    Args:
        images: A NumPy array of image data
        num_distractors: Number of distractor images to show to the "listener"
            alongside the target image
    """
    def __init__(self, images: ndarray, num_distractors: int) -> Dataset:
        super(ImageIdentificationDataset, self).__init__()
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
            'speaker_caps_in': 0,
            'speaker_cap_lens': 0,
            'target': which
        }


def weave_out(caps_out):
    # TODO: figure out why this function exists
    ans = []
    seq_len = max([len(x) for x in caps_out])
    for idx in range(seq_len):
        for sublst in caps_out:
            if idx < len(sublst):
                ans.append(sublst[idx])
    return ans