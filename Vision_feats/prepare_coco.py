import argparse
from builtins import breakpoint
from collections import defaultdict
import json
import os
import random
import torch

def indices_to_string(index_list, index_to_tok):
    # Convert indices to tokens
    tokens = [index_to_tok[x] for x in index_list]
    # Only keep alphanumeric tokens (punctuation won't be tokenized properly)
    tokens = [x for x in tokens if any(char.isalnum() for char in x)]
    # Drop the BOS and EOS tokens
    tokens = tokens[1:-1]
    return ' '.join(tokens)

def dump_captions(directory, name, captions, captions_plus, mode):
    file_name = name + f'_{mode}.jsonl'
    file_name_plus = name + f'_{mode}_plus.jsonl' 
    with open(os.path.join(directory,file_name_plus), 'w+'
        ) as fout:
            for caption_set in captions_plus:
                print(json.dumps(caption_set), file=fout)

    with open(os.path.join(directory,file_name), 'w+'
        ) as fout:
            for caption_set in captions:
                print(json.dumps(caption_set), file=fout)

def process_captions(args):
    caption_dict = defaultdict(list)
    caption_plus_dict = {}
    caption = json.load(open(args.caption_file, 'r'))['annotations']
    images = json.load(open(args.caption_file, 'r'))['images']
    for i in caption:
        caption_dict[str(i['image_id'])].append(i['caption'])
    for i in images:
        caption_plus_dict[str(i['id'])] = i
    # Align the order 
    caption_list = []
    caption_list_plus = []
    with open(args.image_file_name, 'r') as f:
        lines = f.readlines()
        for i in lines:
            id = i.split('_')[-1][:-5].lstrip('0')
            caption_list.append(caption_dict[id])
            caption_list_plus.append(caption_plus_dict[id])
    for index,caption in enumerate(caption_list):
        caption_list_plus[index]['caption'] = caption
    return caption_list, caption_list_plus

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--from_where', required=True, type=str)
    parser.add_argument('--image_feats', required=True, type=str)
    parser.add_argument('--caption_file', required=True, type=str)
    parser.add_argument('--image_file_name', required=True, type=str)
    parser.add_argument('--new_ec_train_images', required=True, type=str)
    parser.add_argument('--new_ec_train_captions', required=True, type=str)
    parser.add_argument('--captioning_captions_base', required=True, type=str)
    parser.add_argument('--captioning_images_base', required=True, type=str)
    parser.add_argument('--image_captioning_size', default=10000, type=int)
    parser.add_argument('--ec_directory', default='Data/ec_finetuning', type=str)
    parser.add_argument('--captioning_directory', default='Data/captioning', type=str)
    parser.add_argument('--seed', default=1, type=int)
    args = parser.parse_args()

    random.seed(args.seed)

    if not os.path.exists(args.ec_directory):
        os.makedirs(args.ec_directory)
    if not os.path.exists(args.captioning_directory):
        os.makedirs(args.captioning_directory)

    all_feats = torch.load(args.image_feats)
    num_examples = all_feats.size(0)
    all_captions, all_captions_plus = process_captions(args)
    permutation = random.sample([i for i in range(num_examples)], num_examples)

    if args.from_where=='val2014':
        # The validation part is only for ec_finetuning validation step
        # Reduce the size of the validation set, leave only 10%
        ec_indices = permutation[:int(0.1*num_examples)] 
        new_ec_feats = all_feats[ec_indices]

        torch.save(
            new_ec_feats, os.path.join(args.ec_directory, args.new_ec_train_images + '_val.pt')
        )

        new_ec_captions = [all_captions[i] for i in ec_indices]
        new_ec_captions_plus = [all_captions_plus[i] for i in ec_indices]
        dump_captions(args.ec_directory , args.captioning_captions_base, 
            new_ec_captions, new_ec_captions_plus, mode='val')
        
        # Dump indices
        with open(
            os.path.join(args.ec_directory, 'val_indices.txt'), 'w+'
        ) as fout:
            for idx in ec_indices:
                print(idx, file=fout)

        return None # End the program here if from validation set

    caption_indices = permutation[:args.image_captioning_size]
    ec_indices = permutation[args.image_captioning_size:]
    caption_train_size = int(0.9 * args.image_captioning_size)
    caption_train_indices = caption_indices[:caption_train_size]
    caption_val_indices = caption_indices[caption_train_size:]

    new_ec_feats = all_feats[ec_indices]
    torch.save(
        new_ec_feats, os.path.join(args.ec_directory, args.new_ec_train_images + '_train.pt')
    )
    captioning_train_feats = all_feats[caption_train_indices]
    captioning_val_feats = all_feats[caption_val_indices]
    torch.save(
        captioning_train_feats,
        os.path.join(
            args.captioning_directory, args.captioning_images_base + '_train.pt'
        )
    )
    torch.save(
        captioning_val_feats,
        os.path.join(
            args.captioning_directory, args.captioning_images_base + '_val.pt'
        )
    )

    new_ec_captions = [all_captions[i] for i in ec_indices]
    new_ec_captions_plus = [all_captions_plus[i] for i in ec_indices]
    dump_captions(args.ec_directory, args.new_ec_train_captions, 
        new_ec_captions, new_ec_captions_plus, mode='train')

    captioning_train_captions = [all_captions[i] for i in caption_train_indices]
    captioning_train_captions_plus = [all_captions_plus[i] for i in caption_train_indices]
    dump_captions(args.captioning_directory, args.captioning_captions_base, 
        captioning_train_captions, captioning_train_captions_plus, mode='train')

    captioning_val_captions = [all_captions[i] for i in caption_val_indices]
    captioning_val_captions_plus = [all_captions_plus[i] for i in caption_val_indices]
    dump_captions(args.captioning_directory, args.captioning_captions_base, 
        captioning_val_captions, captioning_val_captions_plus, mode='val')

    with open(
        os.path.join(args.ec_directory, 'train_indices.txt'), 'w+'
    ) as fout:
        for idx in ec_indices:
            print(idx, file=fout)

    with open(
        os.path.join(args.captioning_directory, 'train_indices.txt'), 'w+'
    ) as fout:
        for idx in caption_train_indices:
            print(idx, file=fout)

    with open(
        os.path.join(args.captioning_directory, 'val_indices.txt'), 'w+'
    ) as fout:
        for idx in caption_val_indices:
            print(idx, file=fout)


if __name__ == "__main__":
    main()