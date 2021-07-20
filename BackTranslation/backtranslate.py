import argparse
import logging
from typing import Dict, List, Any, Union

import torch
import os
import sys
import time
import yaml
from tqdm import tqdm
import random
import numpy as np
from math import ceil

from statistics import mean
from collections import defaultdict, namedtuple
from EC_finetune.agents import CommunicationAgent
from EC_finetune.util import vocab_mask_from_file
from EC_finetune.modelings.modeling_mbart import MBartForConditionalGeneration, BartForConditionalGeneration
from BackTranslation.dataloader import MbartMonolingualDataset
from BackTranslation.constant import LANG_ID_2_LANGUAGE_CODES
from BackTranslation.util import checkpoint_stats2string, translation2string
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import pipeline, AutoTokenizer
from transformers import MBartTokenizer
from datasets import load_dataset
import datasets
# from datasets.wmt19.wmt_utils import WmtConfig
import nlp
from ipdb import set_trace as bp
from wmt19.wmt_utils import WmtConfig

# dataset = datasets.load_dataset('wmt14', 'zh-en')
# dataset = nlp.load_dataset("newstest2017", "zh-en")


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def validation(args, model, tokenizer, source_meta, target_meta):
    # dictionary to store (source->target) statistics needed for ave bleu
    val_metric = args.val_metric
    reference_dataset = args.val_dataset
    accumulative_bleu = 0
    total_translations = 0
    # for source_meta, target_meta in [(args.lang1_meta, args.lang2_meta), (args.lang2_meta, args.lang1_meta)]:
    source_id, source_code, _, source_max_len = list(source_meta)
    target_id, target_code, _, target_max_len = list(target_meta)
    dataloader = torch.utils.data.DataLoader(reference_dataset, batch_size=args.batch_size)
    # args.validation_set_size is a lower bound of how many example used
    num_batch = ceil(args.validation_set_size // args.batch_size)

    for i, batch in enumerate(tqdm(dataloader, total=num_batch)):
        translation_batch = batch["translation"]
        # translation_batch = {k: v.to(args.device) for k, v in translation_batch.items()}
        source_string_batch = translation_batch[source_id]
        reference_batch = translation_batch[target_id]

        if args.val_metric_name == "bleu":
            # "bleu" -> tokenized sentences
            reference_batch = [[tokenizer.tokenize(s)] for s in reference_batch]
        else:
            # "sacrebleu" -> untokenized sentences, sacrebleu's default tokenizers are used
            reference_batch = [[s] for s in reference_batch]

        source_batch = tokenizer.prepare_seq2seq_batch(src_texts=source_string_batch,
                                                       src_lang=source_code,
                                                       tgt_lang=target_code,
                                                       max_length=source_max_len,
                                                       return_tensors="pt")
        source_batch = source_batch.to(args.device)
        translated_ids = model.generate(**source_batch,
                                        decoder_start_token_id=tokenizer.lang_code_to_id[target_code],
                                        max_length=target_max_len)
        if args.val_metric_name == "bleu":
            translations = [tokenizer.convert_ids_to_tokens(s, skip_special_tokens=True) for s in translated_ids]
        else:
            assert args.val_metric_name == "sacrebleu"
            translations = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)
        # print()
        results = val_metric.compute(predictions=translations, references=reference_batch)

        accumulative_bleu += results['score' if args.val_metric_name == "sacrebleu" else 'bleu'] * len(reference_batch)
        total_translations += len(reference_batch)
        if i == num_batch - 1:
            break

    return accumulative_bleu / total_translations


def main(args, source_meta2pack):
    assert len(source_meta2pack) == 2
    source_metas = list(source_meta2pack.keys())
    checkpoint_stats = defaultdict(list)

    for step in range(args.num_steps):
        # we might want to randomly decide the order, because we don't want the model
        # to learn the pattern that we do, e.g., English first and then Japanese second.
        random.shuffle(source_metas)

        # TODO: 4. validate every X
        if step % args.print_every == 0 and args.print_translation:
            translation_results = {args.lang1_id: [], args.lang2_id: []}

        for source_meta in source_metas:
            source_dataloader, tokenizer, source2target_model, target_meta, target2source_model, \
            target2source_model_optimizer = list(source_meta2pack[source_meta])
            source_id, source_code, source_mask, source_max_len = list(source_meta)
            target_id, target_code, target_mask, target_max_len = list(target_meta)
            # bp()

            # # 1. we use source2target_model to generate synthetic text in target language
            # source2target_model.eval()
            # # get a batched string input
            # source_string_batch = next(iter(source_dataloader))
            # source_batch = tokenizer.prepare_seq2seq_batch(src_texts=source_string_batch,
            #                                                src_lang=source_code,
            #                                                tgt_lang=target_code,
            #                                                max_length=source_max_len,
            #                                                return_tensors="pt")
            # source_batch = source_batch.to(args.device)
            # translated_tokens = source2target_model.generate(**source_batch,
            #                                                  decoder_start_token_id=tokenizer.lang_code_to_id[
            #                                                      target_code],
            #                                                  max_length=target_max_len,
            #                                                  lang_mask=target_mask)
            #
            # # turn the predicted subtokens into sentence in string
            # translation = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
            # if step % args.print_every == 0 and args.print_translation:
            #     translation_results[target_id] = translation
            #
            # # 2. we train the target2source_model on the model
            # target2source_model.train()
            # # bp()
            # # Note the synthetic text is the input
            # parallel_batch = tokenizer.prepare_seq2seq_batch(translation,
            #                                                  src_lang=target_code,
            #                                                  tgt_lang=source_code,
            #                                                  tgt_texts=source_string_batch,
            #                                                  max_length=target_max_len,
            #                                                  max_target_length=source_max_len,
            #                                                  return_tensors="pt")
            # parallel_batch = parallel_batch.to(args.device)
            # # bp()
            #
            # output = target2source_model(**parallel_batch)
            # target2source_model_optimizer.zero_grad()
            # output.loss.backward()
            # target2source_model_optimizer.step()
            # checkpoint_stats["loss"].append(output["loss"].detach().cpu().numpy())

            if args.do_validation and step % args.validate_every == 0:
                source2target_model.eval()
                val_score = validation(args, source2target_model, tokenizer, source_meta, target_meta)
                checkpoint_stats[f"val-{args.validation_metric}:{source_id}->{target_id}"].append(val_score)

        if step % args.print_every == 0:
            # bp()
            # TODO: add call to validate() for lang1_2_lang2 model and lang2_2_lang1 model; take average

            checkpoint_average_stats = {}
            for key, value in checkpoint_stats.items():
                checkpoint_average_stats[key] = np.mean(value)
            checkpoint_average_stats[f"ave-val-{args.validation_metric}"] = \
                np.mean([checkpoint_stats[f"val-{args.validation_metric}:{args.lang1_id}->{args.lang2_id}"],
                         checkpoint_stats[f"val-{args.validation_metric}:{args.lang2_id}->{args.lang1_id}"]])
            logger.info(
                checkpoint_stats2string(
                    step, checkpoint_average_stats, 'train'
                )
            )
            checkpoint_stats = defaultdict(list)
            if args.print_translation:
                logger.info(
                    translation2string(
                        translation_results, args.num_printed_translation
                    )
                )

    for source_meta in source_metas:
        source_dataloader, tokenizer, source2target_model, target_meta, target2source_model, _ = \
            list(source_meta2pack[source_meta])
        source_id, source_code, source_mask, source_max_len = list(source_meta)
        target_id, target_code, target_mask, target_max_len = list(target_meta)

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


LangMeta = namedtuple("LangMeta", ["lang_id", "lang_code", "lang_mask", "max_length"])

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
    parser.add_argument('--threshold', type=float, default=0.01,
                        help="Unigram probability threshold to select vocab constraint for"
                             " generation during backtranslation; the higher, the less vocab selected")
    parser.add_argument('--print_translation', action="store_true",
                        help="Whether we want to print backtranslated sentence, for inspection")
    parser.add_argument('--num_printed_translation', type=int, default=3,
                        help="No. of backtranslationed sentences to be printed. "
                             "This number will be capped by the batch size. "
                             "Sentences will be randomly sampled from the *current* batch")
    parser.add_argument('--val_metric_name', type=str, default="bleu")

    # parser.add_argument('--max_length', type=int, default=60)
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
    # bp()
    args.device = device

    tokenizer = MBartTokenizer.from_pretrained(args.model_path)
    # bp()
    lang1_mask = vocab_mask_from_file(tokenizer=tokenizer,
                                      file=args.lang1_vocab_constrain_file,
                                      threshold=args.threshold)
    lang1_mask = lang1_mask.to(args.device)
    print(f"Total valid {args.lang1_id} tokens: {torch.sum(~torch.isinf(lang1_mask))}")

    lang1_valid_tokens = tokenizer.convert_ids_to_tokens((~torch.isinf(lang1_mask)).nonzero(as_tuple=True)[0])
    # print(str(lang1_valid_tokens).encode('utf-8'))
    lang2_mask = vocab_mask_from_file(tokenizer=tokenizer,
                                      file=args.lang2_vocab_constrain_file,
                                      threshold=args.threshold)
    lang2_mask = lang2_mask.to(args.device)
    print(f"Total valid {args.lang2_id} tokens: {torch.sum(~torch.isinf(lang2_mask))}")
    lang2_valid_tokens = tokenizer.convert_ids_to_tokens((~torch.isinf(lang2_mask)).nonzero(as_tuple=True)[0])
    # print(str(lang2_valid_tokens).encode('utf-8'))
    # bp()

    # training_args = torch.load(args.args_path)
    lang1_to_lang2_model = MBartForConditionalGeneration.from_pretrained(args.model_path)
    lang1_to_lang2_model.to(args.device)
    # state_dict = torch.load(args.model_path, map_location=None if torch.cuda.is_available()
    # else torch.device('cpu'))
    # lang1_to_lang2_model.load_state_dict(state_dict)
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

    lang2_to_lang1_model_optimizer = torch.optim.Adam(lang2_to_lang1_model.parameters(), lr=args.lr)
    lang1_to_lang2_model_optimizer = torch.optim.Adam(lang1_to_lang2_model.parameters(), lr=args.lr)

    # lang1_mask = vocab_mask_from_file(tokenizer=tokenizer, file=args.lang1_vocab_constrain_file)
    # lang2_mask = vocab_mask_from_file(tokenizer=tokenizer, file=args.lang2_vocab_constrain_file)

    lang1_meta = LangMeta(lang_id=args.lang1_id, lang_code=args.lang1_code, lang_mask=lang1_mask,
                          max_length=args.lang1_max_len)
    lang2_meta = LangMeta(lang_id=args.lang2_id, lang_code=args.lang2_code, lang_mask=lang2_mask,
                          max_length=args.lang2_max_len)
    args.lang1_meta = lang1_meta
    args.lang2_meta = lang2_meta

    lang1_to_lang2_pack = BackTranslationPack(source_dataloader=lang1_dataloader,
                                              source_tokenizer=tokenizer,
                                              source2target_model=lang1_to_lang2_model,
                                              target_meta=lang2_meta,
                                              target2source_model=lang2_to_lang1_model,
                                              optimizer=lang2_to_lang1_model_optimizer)
    lang2_to_lang1_pack = BackTranslationPack(source_dataloader=lang2_dataloader,
                                              source_tokenizer=tokenizer,
                                              source2target_model=lang2_to_lang1_model,
                                              target_meta=lang1_meta,
                                              target2source_model=lang1_to_lang2_model,
                                              optimizer=lang1_to_lang2_model_optimizer)

    lang_meta2pack = {
        # e.g. (JapanID, JapanCode): [JapanDataset, Japan2EnglishModel, English2JapanModel]
        lang1_meta: lang1_to_lang2_pack,
        lang2_meta: lang2_to_lang1_pack
    }

    if args.do_validation:
        args.val_metric = datasets.load_metric(args.val_metric_name)
        args.val_dataset = load_dataset(args.val_dataset_script, args.lang_pair, split="validation")

    main(args, lang_meta2pack)
    print()
