from builtins import breakpoint
import glob
import os
import csv
import argparse
from pyexpat import model
from turtle import forward
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50
from transformers import BeitFeatureExtractor, BeitModel

class ImageLoader(data.Dataset):
    """The bare minimum class for 
        -loading image data
        -preprossing
        inference purpose only
    """
    def __init__(
        self,
        cur_dir: str,
        model: str,
        do_transform: bool
    ):
        self.transform = transforms_dict[model]
        self.curr_folder = cur_dir
        self.dataset = self.load_imagepaths_with_labels()
        self.num_grayscale = 0
        self.do_transform = do_transform

    def load_imagepaths_with_labels(
        self
    ) -> List[str]:
        """Fetches all images in the dataset.
        """
        img_paths = []
        img_paths = glob.glob(self.curr_folder+'/*.jpg')
        return img_paths

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """Fetches the item at a given index.
        """
        img = None
        path = self.dataset[index]
        with open(path, 'rb') as f:
            img = Image.open(f)
            if img.mode!='RGB':
                self.num_grayscale+=1
                img = img.convert('RGB')
            if self.do_transform:
                img = self.transform(img)
        return img

    def __len__(self) -> int:
        """Returns the number of items in the dataset.
        """
        l = len(self.dataset)
        return l

class MyResNet50(nn.Module):
    def __init__(self):
        """Initialize network layers.
        """
        super().__init__()
        model = resnet50(pretrained=True)
        self.conv_layers = nn.Sequential(*list(model.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net. Just output the feature.
        """
        model_output = self.conv_layers(x)
        return model_output


class MyBeit(nn.Module):
    def __init__(self):
        """Initialize network layers.
        """
        super().__init__()
        #self.feat_ex = BeitFeatureExtractor.from_pretrained("microsoft/beit-large-patch16-224")
        self.model = BeitModel.from_pretrained("microsoft/beit-large-patch16-224")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass with the net. Just output the feature.
        """
        #x = self.feat_ex(x)
        x = {'pixel_values': x}
        model_output = self.model(**x)
        model_output = model_output.last_hidden_state
        return model_output


class VisionEncode(nn.Module):
    def __init__(self, args):
        """A vision encoder for processing images on the fly
        """
        self.feature_extractor = BeitFeatureExtractor.from_pretrained("microsoft/beit-large-patch16-224")
        self.transforms_dict = {
            'resnet': transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]), #Imagenet pretraining default normalization.
                ]),
            # 'beit': transforms.Compose([ #beit's feature extractor would do rest of the job
            #     transforms.Resize(256, 256),
            #     transforms.ToTensor(),
            #     ]),
            'beit': lambda x: torch.from_numpy(self.feature_extractor(x)['pixel_values'][0]) 
        }
        self.models_dict = {
            'resnet': MyResNet50,
            'beit': MyBeit, 
        }
        # Load model
        self.model = self.models_dict[args.model]()
        self.args = args 

    def downsample(self, feats):
        """
        reduce the number of tokens in the image feature.
        """
        batch_size = feats.shape[0]
        s_feats = feats[:, 0].unsqueeze(1)
        o_feats = feats[:, 1:]
        num_patch = int(np.sqrt((o_feats.shape[1])))
        embed_size = s_feats.shape[-1]
        o_feats = o_feats.transpose(2,1)
        o_feats = o_feats.reshape(batch_size, embed_size, num_patch, num_patch)
        m = nn.AvgPool2d(self.args.kernel_size, stride=self.args.kernel_size)
        o_feats = m(o_feats) # (batch_size, embed_size, new_num_patch, new_num_patch)
        o_feats = o_feats.reshape(batch_size, embed_size, -1)
        o_feats = o_feats.transpose(2, 1)
        feats = torch.cat([s_feats, o_feats], dim=1)
        return feats

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.model(x)
        feats = feats.squeeze()
        if self.args.do_downsample:
            feats = self.downsample(feats, self.args)
        return feats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='val2014', type=str)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--model', default='resnet', type=str)
    parser.add_argument('--output_dir', default='./', type=str)
    parser.add_argument('--output_name', default='val_feats.pt', type=str)
    parser.add_argument('--output_id_name', default='val_feats_id.txt', type=str)
    parser.add_argument('--do_downsample', action='store_true')
    parser.add_argument('--kernel_size',default=4, type=int)
    parser.add_argument('--do_transform', action='store_true')
    args = parser.parse_args()

    device = torch.device("cuda:0")
    # Load dataset
    dataset = ImageLoader(args.data_dir, args.model, args.do_transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    # Load model
    model = VisionEncode(args) 
    model.to(device)
    model.eval()

    feats_list = []
    with torch.no_grad():
        for x in tqdm(loader):
            x = x.to(device)
            feats = model(x)
            feats_list.append(feats.cpu())
        
    all_feats = torch.cat(feats_list, dim=0)
    all_ids = [id.split('/')[-1] for id in dataset.dataset]

    # Save feature and id
    print(f'There are {dataset.num_grayscale} gray scale images being converted.')
    torch.save(all_feats, os.path.join(args.output_dir, args.output_name))

    with open(os.path.join(args.output_dir, args.output_id_name), 'w') as f:
        for id in all_ids:
            f.write(id)
            f.write('\n')

if __name__ == "__main__":
    main()
