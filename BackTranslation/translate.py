import argparse
import logging
import os
import sys
from argparse import Namespace

import torch
import tqdm
import yaml
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import MBartTokenizer

from BackTranslation.constant import LANG_ID_2_LANGUAGE_CODES
from BackTranslation.util import set_seed
from EC_finetune.modelings.modeling_mbart import MBartForConditionalGeneration


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
        source_string_batch = batch['text']
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
            max_length=args.target_max_len
        )
        translation_str = tokenizer.batch_decode(
            translated_ids, skip_special_tokens=True
        )
        translation_lines += [line for line in translation_str]

    with open(args.translation_file, 'w+') as fout:
        for line in translation_lines:
            print(line, file=fout)


def main():
    """
    Script to use an MBart model to translate a source dataset into a target
    language
    """
    # Configure the logger (boilerplate)
    logger = logging.getLogger(__name__)
    out_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(out_handler)
    message_format = '%(asctime)s - %(message)s'
    date_format = '%m-%d-%y %H:%M:%S'
    out_handler.setFormatter(logging.Formatter(message_format, date_format))
    out_handler.setLevel(logging.INFO)
    logger.addHandler(out_handler)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="MBart Translation Script")

    parser.add_argument('--config', type=str)

    args = parser.parse_args()
    args_dict = vars(args)

    with open(args_dict['config'], 'r') as config_file:
        args_dict.update(yaml.load(config_file, Loader=yaml.SafeLoader))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # set random seed
    set_seed(args)

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    tokenizer = MBartTokenizer.from_pretrained(args.model_path)

    model = MBartForConditionalGeneration.from_pretrained(args.model_path)
    model.to(args.device)

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

    source_dataset = load_dataset(
        "text", data_files=os.path.join(args.data_dir, args.source_data_file)
    )
    source_dataloader = DataLoader(
        source_dataset, batch_size=args.batch_size, shuffle=True
    )

    args.translation_file = os.path.join(
        args.output_dir, args.translation_filename
    )

    translate(args, model, source_dataloader, tokenizer)


if __name__ == "__main__":
    main()
