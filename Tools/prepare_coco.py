import argparse
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_file', required=True, type=str)
    parser.add_argument('--caption_file', required=True, type=str)
    parser.add_argument('--new_ec_train_images', required=True, type=str)
    parser.add_argument('--new_ec_train_captions', required=True, type=str)
    parser.add_argument('--captioning_captions_base', required=True, type=str)
    parser.add_argument('--captioning_images_base', required=True, type=str)
    parser.add_argument('--index2tok_dictionary', required=True, type=str)
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

    all_feats = torch.load(args.image_file)
    num_examples = all_feats.size(0)

    all_captions = torch.load(args.caption_file)
    index_to_tok = torch.load(args.index2tok_dictionary)
    all_captions = [
        [indices_to_string(sentence, index_to_tok) for sentence in example]
        for example in all_captions
    ]

    permutation = random.sample([i for i in range(num_examples)], num_examples)
    caption_indices = permutation[:args.image_captioning_size]
    ec_indices = permutation[args.image_captioning_size:]
    caption_train_size = int(0.9 * args.image_captioning_size)
    caption_train_indices = caption_indices[:caption_train_size]
    caption_val_indices = caption_indices[caption_train_size:]

    new_ec_feats = all_feats[ec_indices]
    torch.save(
        new_ec_feats, os.path.join(args.ec_directory, args.new_ec_train_images)
    )
    captioning_train_feats = all_feats[caption_train_indices]
    captioning_val_feats = all_feats[caption_val_indices]
    torch.save(
        captioning_train_feats,
        os.path.join(
            args.captioning_directory, args.captioning_images_base + '_train'
        )
    )
    torch.save(
        captioning_val_feats,
        os.path.join(
            args.captioning_directory, args.captioning_images_base + '_val'
        )
    )

    new_ec_captions = [all_captions[i] for i in ec_indices]
    with open(
        os.path.join(args.ec_directory, args.new_ec_train_captions), 'w+'
    ) as fout:
        for caption_set in new_ec_captions:
            print(json.dumps(caption_set), file=fout)

    captioning_train_captions = [all_captions[i] for i in caption_train_indices]
    with open(
        os.path.join(
            args.captioning_directory,
            args.captioning_captions_base + '_train.jsonl'
        ), 'w+'
    ) as fout:
        for caption_set in captioning_train_captions:
            print(json.dumps(caption_set), file=fout)

    captioning_val_captions = [all_captions[i] for i in caption_val_indices]
    with open(
        os.path.join(
            args.captioning_directory,
            args.captioning_captions_base + '_val.jsonl'
        ), 'w+'
    ) as fout:
        for caption_set in captioning_val_captions:
            print(json.dumps(caption_set), file=fout)

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
