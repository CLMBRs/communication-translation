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
from collections import defaultdict, namedtuple
from EC_finetune.agents import CommunicationAgent
from EC_finetune.util import vocab_mask_from_file
from EC_finetune.modelings.modeling_mbart import MBartForConditionalGeneration
from BackTranslation.dataloader import MbartMonolingualDataset
from BackTranslation.constant import LANG_ID_2_LANGUAGE_CODES
from BackTranslation.util import checkpoint_stats2string
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import pipeline, AutoTokenizer
from transformers import BartForConditionalGeneration, MBartTokenizer
import datasets
from datasets.wmt19.wmt_utils import WmtConfig

# dataset = datasets.load_dataset('wmt14', 'zh-en')
config = WmtConfig(
    version="0.0.1",
    language_pair=("zh", "en"),
    subsets={
        datasets.Split.VALIDATION: ["newsdev2019"],
    },
)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def generate_synthetic_dataset(args, source_meta2pack):
    assert len(source_meta2pack) == 2
    source_metas = list(source_meta2pack.keys())
    checkpoint_stats = defaultdict(list)

    for step in range(args.num_steps):
        # we might want to randomly decide the order, because we don't want the model
        # to learn the pattern that we do, e.g., English first and then Japanese second.
        random.shuffle(source_metas)

        # TODO: 1. have support to constrain generation!!!
        # TODO: 3. have support to clip_grad_norm?
        # TODO: 4. validate every X

        for source_meta in source_metas:
            source_dataloader, tokenizer, source2target_model, target_meta, target2source_model, \
            target2source_model_optimizer = list(source_meta2pack[source_meta])
            source_id, source_code, source_mask = list(source_meta)
            target_id, target_code, target_mask = list(target_meta)

            # 1. we use source2target_model to generate synthetic text in target language
            source2target_model.eval()
            # get a batched string input
            source_string_batch = next(iter(lang1_dataloader))
            # tokenize the string batch
            source_batch = tokenizer.prepare_seq2seq_batch(src_texts=source_string_batch,
                                                           src_lang=source_code,
                                                           return_tensors="pt")
            # generate the synthetic target sentence
            max_len = source_batch["input_ids"].shape[1]
            # not necessrily using gumbel_generate  consider beam search
            translated_tokens = source2target_model.gumbel_generate(**source_batch,
                                                                    decoder_start_token_id=tokenizer.lang_code_to_id[
                                                                    target_code],
                                                                    max_length=max_len,
                                                                    lang_mask=target_mask)
            # turn the predicted subtokens into sentence in string
            translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)

            # 2. we train the target2source_model on the model
            target2source_model.train()
            # Note the synthetic output is the input
            parallel_batch = tokenizer.prepare_seq2seq_batch(translation,
                                                             src_lang=target_code,
                                                             tgt_lang=source_code,
                                                             tgt_texts=source_string_batch,
                                                             return_tensors="pt")

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

    for source_meta in source_metas:
        source_dataloader, tokenizer, source2target_model, target_meta, target2source_model, _ = \
            list(source_meta2pack[source_meta])
        source_id, source_code = list(source_meta)
        target_id, target_code = list(target_meta)

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
        else:
            torch.save(
                source2target_model.state_dict(), os.path.join(args.output_dir, f"{source_id}2{target_id}", "model.pt")
            )
            # Good practice: save your training arguments together
            # with the trained model
            torch.save(
                args,
                os.path.join(args.output_dir, f"{source_id}2{target_id}", "training_args.bin")
            )

            torch.save(
                target2source_model.state_dict(), os.path.join(args.output_dir, f"{target_id}2{source_id}", "model.pt")
            )
            # Good practice: save your training arguments together
            # with the trained model
            torch.save(
                args,
                os.path.join(args.output_dir, f"{target_id}2{source_id}", "training_args.bin")
            )
        # we just save once
        break


LangMeta = namedtuple("LangMeta", ["lang_id", "lang_code", "lang_mask"])

BackTranslationPack = namedtuple("BackTranslationPack",
                                 ["source_dataloader",
                                  "source_tokenizer",
                                  "source2target_model",
                                  "target_meta",
                                  "target2source_model",
                                  "optimizer",
                                  ])

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

    lang1_dataset = MbartMonolingualDataset(
        source_file=os.path.join(args.data_dir, args.lang1_data_file),
        tokenizer=tokenizer,
        lang_code=LANG_ID_2_LANGUAGE_CODES[args.lang1_id],
    )
    lang2_dataset = MbartMonolingualDataset(
        source_file=os.path.join(args.data_dir, args.lang2_data_file),
        tokenizer=tokenizer,
        lang_code=LANG_ID_2_LANGUAGE_CODES[args.lang2_id],
    )
    lang1_dataloader = DataLoader(lang1_dataset, batch_size=args.batch_size, shuffle=True)
    lang2_dataloader = DataLoader(lang2_dataset, batch_size=args.batch_size, shuffle=True)
    # collate_fn=lambda x: pad_sequence(x, batch_first=True, padding_value=tokenizer.vocab['<pad>']))

    lang2_to_lang1_model_optimizer = torch.optim.Adam(lang2_to_lang1_model.model.parameters(), lr=args.lr)
    lang1_to_lang2_model_optimizer = torch.optim.Adam(lang1_to_lang2_model.model.parameters(), lr=args.lr)

    lang1_mask = vocab_mask_from_file(tokenizer=tokenizer, file=args.lang1_vocab_constrain_file)
    lang2_mask = vocab_mask_from_file(tokenizer=tokenizer, file=args.lang2_vocab_constrain_file)

    lang1_meta = LangMeta(lang_id=args.lang1_id, lang_code=args.lang1_code, lang_mask=lang1_mask)
    lang2_meta = LangMeta(lang_id=args.lang2_id, lang_code=args.lang2_code, lang_mask=lang2_mask)

    lang1_to_lang2_pack = BackTranslationPack(source_dataloader=lang1_dataloader,
                                              source_tokenizer=tokenizer,
                                              source2target_model=lang1_to_lang2_model.model,
                                              target_meta=lang2_meta,
                                              target2source_model=lang2_to_lang1_model.model,
                                              optimizer=lang2_to_lang1_model_optimizer)
    lang2_to_lang1_pack = BackTranslationPack(source_dataloader=lang2_dataloader,
                                              source_tokenizer=tokenizer,
                                              source2target_model=lang2_to_lang1_model.model,
                                              target_meta=lang1_meta,
                                              target2source_model=lang1_to_lang2_model.model,
                                              optimizer=lang1_to_lang2_model_optimizer)

    lang_meta2pack = {
        # e.g. (JapanID, JapanCode): [JapanDataset, Japan2EnglishModel, English2JapanModel]
        lang1_meta: lang1_to_lang2_pack,
        lang2_meta: lang2_to_lang1_pack
    }
    generate_synthetic_dataset(args, lang_meta2pack)

    print()
