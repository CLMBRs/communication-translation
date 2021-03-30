import argparse
import logging
import torch
import os
import sys
import time
import yaml
from tqdm import tqdm
import random
import numpy as np

from statistics import mean
from collections import defaultdict
from EC_finetune.agents import CommunicationAgent
from BackTranslation.dataloader import MbartMonolingualDataset
from BackTranslation.constant import LANG_ID_2_LANGUAGE_CODES
from BackTranslation.util import checkpoint_stats2string
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import pipeline, AutoTokenizer
from transformers import BartForConditionalGeneration, MBartTokenizer


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def generate_synthetic_dataset(args, lang_code2pack):
    assert len(lang_code2pack) == 2
    lang_codes = list(lang_code2pack.keys())
    checkpoint_stats = defaultdict(list)

    for step in range(args.num_steps):
        # we might want to randomly decide the order, because we don't want the model
        # to learn the pattern that we do, e.g., English first and then Japanese second.
        random.shuffle(lang_codes)

        # TODO: 1. have support to constrain generation!!!
        # TODO: 3. have support to clip_grad_norm
        # TODO: 4. validate every X

        for source_id, source_code in lang_codes:
            source_dataloader, tokenizer, source2target_model, target_code, target2source_model, target2source_model_optimizer = lang_code2pack[(source_id, source_code)]
            # 1. we use source2target_model to generate synthetic text in target language
            source2target_model.eval()
            # get a batched string input
            source_string_batch = next(iter(lang1_dataloader))
            # tokenize the string batch
            source_batch = tokenizer.prepare_seq2seq_batch(src_texts=source_string_batch, src_lang=source_code, return_tensors="pt")
            # generate the synthetic target sentence
            translated_tokens = source2target_model.generate(**source_batch,
                                                             decoder_start_token_id=tokenizer.lang_code_to_id[target_code],
                                                             max_length=source_batch["input_ids"].shape[1])
            # turn the predicted subtokens into sentence in string
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

            # 2. we train the target2source_model on the model
            target2source_model.train()
            # Note the synthetic output is the input
            parallel_batch = tokenizer.prepare_seq2seq_batch(translation, src_lang=target_code, tgt_lang=source_code,
                                                             tgt_texts=source_string_batch, return_tensors="pt")

            output = target2source_model(**parallel_batch)
            target2source_model_optimizer.zero_grad()
            output.loss.backward()
            target2source_model_optimizer.step()
            for key in ["loss"]:
                checkpoint_stats[key].append(output[key])

        if step % args.print_every == 0:
            checkpoint_average_stats = {}
            for key, value in checkpoint_stats.items():
                checkpoint_average_stats[key] = mean(value)
            logger.info(
                checkpoint_stats2string(
                    step, checkpoint_average_stats, 'train'
                )
            )
            checkpoint_stats = defaultdict(list)

    for source_id, source_code in lang_codes:
        source_dataloader, tokenizer, source2target_model, _, target2source_model, _ = \
        lang_code2pack[(source_id, source_code)]

        # Save the general part of the model
        if args.models_shared:
            torch.save(
                source2target_model.state_dict(), os.path.join(args.output_dir, "model.pt")
            )
            # Good practice: save your training arguments together
            # with the trained model
            torch.save(
                args,
                os.path.join(args.output_dir, "training_args.bin")
            )
        else





if __name__ == "__main__":
    # Configure the logger (boilerplate)
    logger = logging.getLogger(__name__)
    out_handler = logging.StreamHandler(sys.stdout)
    message_format = '%(asctime)s - %(message)s'
    date_format = '%m-%d-%y %H:%M:%S'
    out_handler.setFormatter(logging.Formatter(message_format, date_format))
    out_handler.setLevel(logging.INFO)
    logger.addHandler(out_handler)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Backtranslation Engine")

    parser.add_argument('--backtranslated_dir', type=str, default="Output/")
    # parser.add_argument('--source_dir', type=str, default="./Data/BackTranslate")
    parser.add_argument('--config', type=str)
    args = parser.parse_args()
    args_dict = vars(args)

    with open(args_dict['config'], 'r') as config_file:
        args_dict.update(yaml.load(config_file, Loader=yaml.SafeLoader))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # set random seed
    set_seed(args)

    # Start the clock for the beginning of the main function
    start_time = time.time()
    logging.info('Entering main run script')

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    tokenizer = MBartTokenizer.from_pretrained("facebook/mbart-large-cc25")

    training_args = torch.load(args.args_path)
    lang1_to_lang2_model = CommunicationAgent(training_args)
    state_dict = torch.load(args.model_path, map_location=None if torch.cuda.is_available()
    else torch.device('cpu'))
    lang1_to_lang2_model.load_state_dict(state_dict)
    if args.models_shared:
        lang2_to_lang1_model = lang1_to_lang2_model  # CommunicationAgent(training_args)
    else:
        raise NotImplementedError("Not yet")
    # state_dict = torch.load(args.model_path, map_location=None if torch.cuda.is_available()
    # else torch.device('cpu'))
    # target_to_source_model.load_state_dict(state_dict)

    assert args.lang1_id in LANG_ID_2_LANGUAGE_CODES and args.lang2_id in LANG_ID_2_LANGUAGE_CODES
    args.lang1_code = LANG_ID_2_LANGUAGE_CODES[args.lang1_id]
    args.lang2_code = LANG_ID_2_LANGUAGE_CODES[args.lang2_id]

    lang1_dataset = MbartMonolingualDataset(os.path.join(args.data_dir, args.lang1_data_file),
                                            tokenizer, LANG_ID_2_LANGUAGE_CODES[args.lang1_id])
    lang2_dataset = MbartMonolingualDataset(
        os.path.join(args.data_dir, args.lang2_data_file),
        tokenizer, LANG_ID_2_LANGUAGE_CODES[args.lang2_id]
    )
    lang1_dataloader = DataLoader(lang1_dataset, batch_size=args.batch_size, shuffle=True)
    lang2_dataloader = DataLoader(lang2_dataset, batch_size=args.batch_size, shuffle=True)  # collate_fn=lambda x: pad_sequence(x, batch_first=True, padding_value=tokenizer.vocab['<pad>']))

    lang2_to_lang1_model_optimizer = torch.optim.Adam(lang2_to_lang1_model.model.parameters(), lr=args.lr)
    lang1_to_lang2_model_optimizer = torch.optim.Adam(lang1_to_lang2_model.model.parameters(), lr=args.lr)

    lang_code2pack = {
        # e.g. (JapanID, JapanCode): [JapanDataset, Japan2EnglishModel, English2JapanModel]
        (args.lang1_id, args.lang1_code): [lang1_dataloader, tokenizer, lang1_to_lang2_model.model, args.lang2_code, lang2_to_lang1_model.model, lang2_to_lang1_model_optimizer],
        (args.lang2_id, args.lang2_code): [lang2_dataloader, tokenizer, lang2_to_lang1_model.model, args.lang1_code, lang1_to_lang2_model.model, lang1_to_lang2_model_optimizer]
    }
    generate_synthetic_dataset(args, lang_code2pack)

    print()
