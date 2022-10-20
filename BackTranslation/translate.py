import argparse
import logging
import os
import sys
from argparse import Namespace

import torch
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartTokenizer

from BackTranslation.constant import LANG_ID_2_LANGUAGE_CODES
from EC_finetune.modelings.modeling_mbart import MBartForConditionalGeneration
from Util.util import create_logger, set_seed

def translate(
    args: Namespace, model: MBartForConditionalGeneration,
    dataloader: DataLoader, tokenizer: MBartTokenizer
):
    """
    Use an MBart model to translate a source dataset into a target language

    Args:
        args: a Namespace of script arguments. Must include `device`,
            `source_id`, `target_id`, `source_code`, `target_code`,
            `source_max_len`, `target_max_len`, and `translation_file`
        model: the MBart model for conducting translation
        dataloader: the dataloader over the source dataset
        tokenizer: the tokenizer for the MBart model
    """
    translation_lines = []

    for batch in tqdm(
        dataloader, desc=f"translate:{args.source_id}->{args.target_id}"
    ):
        if args.is_translation_dataset:
            translation_batch = batch["translation"]
            source_string_batch = [
                x for x in translation_batch[args.source_id] if x.strip() != ''
            ]
        else:
            source_string_batch = batch['text']
            source_string_batch = [
                x for x in source_string_batch if x.strip() != ''
            ]

        source_batch = tokenizer.prepare_seq2seq_batch(
            src_texts=source_string_batch,
            src_lang=args.source_code,
            tgt_lang=args.target_code,
            max_length=args.source_max_len,
            return_tensors="pt"
        )
        source_batch = source_batch.to(args.device)

        translated_ids = model.generate(
            **source_batch,
            decoder_start_token_id=tokenizer.lang_code_to_id[args.target_code],
            num_beams=args.num_beams,
            max_length=args.target_max_len
        )
        translation_str = tokenizer.batch_decode(
            translated_ids, skip_special_tokens=True
        )
        translation_lines += [line for line in translation_str]

    with open(args.translation_file, 'w+', encoding='utf-8') as fout:
        for line in translation_lines:
            print(line, file=fout)


def main():
    """
    Script to use an MBart model to translate a source dataset into a target
    language
    """

    logger = create_logger(name="translate")

    parser = argparse.ArgumentParser(description="MBart Translation Script")

    parser.add_argument('--config', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_dir', type=str)

    args = parser.parse_args()
    args_dict = vars(args)

    with open(args_dict['config'], 'r') as config_file:
        args_dict.update(yaml.load(config_file, Loader=yaml.SafeLoader))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # set random seed
    set_seed(args.seed, args.n_gpu)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    tokenizer = MBartTokenizer.from_pretrained(
        args.model_path, local_files_only=True
    )

    model = MBartForConditionalGeneration.from_pretrained(
        args.model_path, local_files_only=True
    )
    model.to(args.device)
    model.eval()

    assert (
        args.source_id in LANG_ID_2_LANGUAGE_CODES and
        args.target_id in LANG_ID_2_LANGUAGE_CODES
    )
    args.source_code = LANG_ID_2_LANGUAGE_CODES[args.source_id]
    args.target_code = LANG_ID_2_LANGUAGE_CODES[args.target_id]
    logger.info(
        f"Source language code: {args.source_code},"
        f" target language code: {args.target_code}"
    )

    if getattr(args, 'dataset_script', False):
        source_dataset = load_dataset(
            args.dataset_script, args.lang_pair, split=args.dataset_split
        )
        args.is_translation_dataset = True
    elif getattr(args, 'source_data_file', False):
        source_dataset = load_dataset(
            "text", data_files=args.source_data_file
        )['train']
        args.is_translation_dataset = False
    else:
        raise ValueError(
            "Configuration must include either `dataset_script` or"
            " `source_data_file` with which to load the source data"
        )
    source_dataloader = DataLoader(
        source_dataset, batch_size=args.batch_size, shuffle=False
    )

    args.translation_file = os.path.join(
        args.output_dir, args.translation_filename
    )

    translate(args, model, source_dataloader, tokenizer)


if __name__ == "__main__":
    main()
