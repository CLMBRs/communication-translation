import argparse
import logging
from statistics import mean

import torch
import os
import sys
import time
import yaml
from tqdm import tqdm
import random
import numpy as np
from math import ceil, isinf

from collections import defaultdict, namedtuple
from EC_finetune.util import vocab_constraint_from_file
from EC_finetune.modelings.modeling_mbart import MBartForConditionalGeneration
from BackTranslation.constant import LANG_ID_2_LANGUAGE_CODES
from BackTranslation.util import translation2string
from torch.utils.data import DataLoader
from transformers import MBartTokenizer
from datasets import load_dataset
import datasets


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def statbar_string(stat_dict: dict) -> str:
    """
    Return a printable "statbar" string from a dictionary of named statistics
    """
    stat_items = []
    for key, value in stat_dict.items():
        stat_items.append(f"{key} {value}")
    return ' | '.join(stat_items)


def get_next_batch(dataloader, data_iter):
    try:
        data = next(data_iter)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        data_iter = iter(dataloader)
        data = next(data_iter)
    return data


def write_validation_splits(
    args, source_id, target_id
):
    reference_dataset = args.val_dataset
    dataloader = torch.utils.data.DataLoader(
        reference_dataset, batch_size=args.batch_size
    )
    source_lines = []
    target_lines = []
    num_batch = ceil(args.validation_set_size / args.batch_size)
    for i, batch in enumerate(dataloader):
        if i == num_batch:
            break
        translation_batch = batch["translation"]
        source_lines += [line for line in translation_batch[source_id]]
        target_lines += [line for line in translation_batch[target_id]]
    source_filename = f"{args.lang_pair}.{source_id}.val"
    target_filename = f"{args.lang_pair}.{target_id}.val"
    source_file = os.path.join(args.output_dir, source_filename)
    target_file = os.path.join(args.output_dir, target_filename)
    with open(source_file, 'w+') as f:
        for line in source_lines:
            print(line, file=f)
    with open(target_file, 'w+') as f:
        for line in target_lines:
            print(line, file=f)


def save_model(args, backtranslation_pack, saved_model_name):
    source_meta, target_meta = backtranslation_pack.metas
    source2target_model, target2source_model = backtranslation_pack.models    
    source_id, _, _ = list(source_meta)
    target_id, _, _ = list(target_meta)

    # Save the general part of the model
    if args.models_shared:
        torch.save(
            source2target_model.state_dict(),
            os.path.join(args.output_dir, saved_model_name)
        )
    else:
        torch.save(
            source2target_model.state_dict(),
            os.path.join(
                args.output_dir, f"{source_id}2{target_id}",
                saved_model_name
            )
        )
        torch.save(
            target2source_model.state_dict(),
            os.path.join(
                args.output_dir, f"{target_id}2{source_id}",
                saved_model_name
            )
        )


def get_translation_score(
    args,
    model,
    tokenizer,
    source_meta,
    target_meta
):
    val_metric = args.val_metric
    reference_dataset = args.val_dataset
    cumulative_score = 0
    total_translations = 0
    translation_lines = []

    source_id, source_code, source_max_len = list(source_meta)
    target_id, target_code, target_max_len = list(target_meta)
    dataloader = torch.utils.data.DataLoader(
        reference_dataset, batch_size=args.batch_size
    )
    num_batch = ceil(args.validation_set_size / args.batch_size)

    for i, batch in enumerate(
        tqdm(
            dataloader,
            total=num_batch,
            desc=f"val-{args.val_metric_name}:{source_id}->{target_id}"
        )
    ):
        if i == num_batch:
            break
        translation_batch = batch["translation"]
        source_string_batch = translation_batch[source_id]
        reference_batch_str = translation_batch[target_id]

        if args.val_metric_name == "bleu":
            # "bleu" -> tokenized sentences
            reference_batch = [
                [tokenizer.tokenize(s)] for s in reference_batch_str
            ]
        else:
            # "sacrebleu" -> untokenized sentences, sacrebleu's default tokenizers are used
            reference_batch = [[s] for s in reference_batch_str]

        source_batch = tokenizer.prepare_seq2seq_batch(
            src_texts=source_string_batch,
            src_lang=source_code,
            tgt_lang=target_code,
            max_length=source_max_len,
            return_tensors="pt"
        )
        source_batch = source_batch.to(args.device)
        translated_ids = model.generate(
            **source_batch,
            decoder_start_token_id=tokenizer.lang_code_to_id[target_code],
            max_length=target_max_len
        )
        if args.val_metric_name == "bleu":
            translations = [
                tokenizer.convert_ids_to_tokens(s, skip_special_tokens=True)
                for s in translated_ids
            ]
        else:
            assert args.val_metric_name == "sacrebleu"
            translations = tokenizer.batch_decode(
                translated_ids, skip_special_tokens=True
            )
        translation_str = tokenizer.batch_decode(
            translated_ids, skip_special_tokens=True
        )
        translation_lines += [line for line in translation_str]
        
        results = val_metric.compute(
            predictions=translations, references=reference_batch
        )

        score = results['score' if args.val_metric_name == "sacrebleu" else 'bleu']
        cumulative_score += score * len(reference_batch)
        total_translations += len(reference_batch)

    return cumulative_score / total_translations, translation_lines


def evaluate(args, backtranslation_pack, best_score, patience_count, step):
    source2target_model, target2source_model = backtranslation_pack.models
    tokenizer = backtranslation_pack.tokenizer
    source_meta, target_meta = backtranslation_pack.metas
    source_id, _, _ = list(source_meta)
    target_id, _, _ = list(target_meta)
    
    source2target_model.eval()
    target2source_model.eval()

    source2target_score, source2target_translations = get_translation_score(
        args, source2target_model, tokenizer, source_meta, target_meta
    )
    target2source_score, target2source_translations = get_translation_score(
        args, target2source_model, tokenizer, target_meta, source_meta
    )

    mean_score = round(mean([source2target_score, target2source_score]), 4)
    stats = {
        'step': step,
        'mode': 'validation',
        f'{target_id} bleu': round(source2target_score, 4),
        f'{source_id} bleu': round(target2source_score, 4),
        'mean bleu': mean_score
    }
    logger.info(statbar_string(stats))

    if mean_score > best_score or isinf(best_score):
        # if we encounter a better model, we restart patience counting
        # and save the model
        patience_count = 0
        best_score = mean_score
        save_model(args, backtranslation_pack, saved_model_name="best.pt")
        source_filename = f"{args.lang_pair}.{source_id}.val.{target_id}"
        target_filename = f"{args.lang_pair}.{target_id}.val.{source_id}"
        source_file = os.path.join(args.output_dir, source_filename)
        target_file = os.path.join(args.output_dir, target_filename)
        with open(source_file, 'w+') as f:
            for line in source2target_translations:
                print(line, file=f)
        with open(target_file, 'w+') as f:
            for line in target2source_translations:
                print(line, file=f)
    else:
        if step >= args.early_stop_start_time:
            # we start counting the early stopping after some 'warmup period'
            patience_count += 1

    return best_score, patience_count


def main(args, backtranslation_pack):
    checkpoint_stats = defaultdict(list)
    best_score = float("-inf")
    patience_count = 0

    for step in range(args.num_steps):
        # we might want to randomly decide the order, because we don't want the model
        # to learn the pattern that we do, e.g., English first and then Japanese second.

        if step % args.print_every == 0 and args.print_translation:
            translation_results = {args.lang1_id: [], args.lang2_id: []}

        for source in random.sample([0, 1], 2):
            target = np.abs(1 - source)
            
            source_dataloader = backtranslation_pack.dataloaders[source]
            source_data_iter = backtranslation_pack.iterators[source]
            tokenizer = backtranslation_pack.tokenizer
            source2target_model = backtranslation_pack.models[source]
            target2source_model = backtranslation_pack.models[target]
            target_vocab_constraint = backtranslation_pack.vocab_constraints[target]
            target2source_optimizer = backtranslation_pack.optimizers[target]
            source_meta = backtranslation_pack.metas[source]
            target_meta = backtranslation_pack.metas[target]

            source_id, source_code, source_max_len = list(source_meta)
            target_id, target_code, target_max_len = list(target_meta)

            if step == 0:
                write_validation_splits(args, source_id, target_id)

            # 1. we use source2target_model to generate synthetic text in target language
            source2target_model.eval()
            # get a batched string input
            source_string_batch = get_next_batch(
                source_dataloader, source_data_iter
            )["text"]
            source_batch = tokenizer.prepare_seq2seq_batch(
                src_texts=source_string_batch,
                src_lang=source_code,
                tgt_lang=target_code,
                max_length=source_max_len,
                return_tensors="pt"
            )
            source_batch = source_batch.to(args.device)
            translated_tokens = source2target_model.generate(
                **source_batch,
                decoder_start_token_id=tokenizer.lang_code_to_id[target_code],
                max_length=target_max_len,
                lang_mask=target_vocab_constraint
            )

            # turn the predicted subtokens into sentence in string
            translation = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )
            if step % args.print_every == 0 and args.print_translation:
                translation_results[target_id] = list(
                    zip(source_string_batch, translation)
                )

            # 2. we train the target2source_model on the model
            target2source_model.train()
            # Note the synthetic text is the input
            parallel_batch = tokenizer.prepare_seq2seq_batch(
                translation,
                src_lang=target_code,
                tgt_lang=source_code,
                tgt_texts=source_string_batch,
                max_length=target_max_len,
                max_target_length=source_max_len,
                return_tensors="pt"
            )
            parallel_batch = parallel_batch.to(args.device)

            output = target2source_model(**parallel_batch)
            target2source_optimizer.zero_grad()
            output.loss.backward()
            target2source_optimizer.step()
            checkpoint_stats["loss"].append(
                output["loss"].detach().cpu().item()
            )

        if args.do_validation and step % args.validate_every == 0:
            best_score, patience_count = evaluate(
                args, backtranslation_pack, best_score, patience_count, step
            )
            if patience_count > args.patience:
                break

        if step % args.print_every == 0:
            checkpoint_average_stats = {}
            checkpoint_average_stats['step'] = step
            checkpoint_average_stats['mode'] = "train"
            for key, value in checkpoint_stats.items():
                checkpoint_average_stats[key] = np.mean(value)
            logger.info(
                statbar_string(checkpoint_average_stats)
            )
            checkpoint_stats = defaultdict(list)
            if args.print_translation:
                logger.info(
                    translation2string(
                        translation_results, args.num_printed_translation
                    )
                )

    save_model(args, backtranslation_pack, saved_model_name="last.pt")


LangMeta = namedtuple("LangMeta", ["lang_id", "lang_code", "max_length"])

BackTranslationPack = namedtuple(
    "BackTranslationPack", [
        "dataloaders",
        "iterators",
        "tokenizer",
        "models",
        "metas",
        "vocab_constraints",
        "optimizers",
    ]
)

if __name__ == "__main__":
    # Configure the logger (boilerplate)
    logger = logging.getLogger(__name__)
    out_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(out_handler)
    message_format = '%(asctime)s - %(message)s'
    date_format = '%m-%d-%y %H:%M:%S'
    # logger.Formatter(message_format, date_format)
    out_handler.setFormatter(logging.Formatter(message_format, date_format))
    out_handler.setLevel(logging.INFO)
    logger.addHandler(out_handler)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description="Backtranslation Engine")

    parser.add_argument('--backtranslated_dir', type=str, default="Output/")
    # parser.add_argument('--source_dir', type=str, default="./Data/BackTranslate")
    parser.add_argument('--config', type=str)
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.01,
        help="Unigram probability threshold to select vocab constraint for"
        " generation during backtranslation; the higher, the less vocab selected"
    )
    parser.add_argument(
        '--print_translation',
        action="store_true",
        help="Whether we want to print backtranslated sentence, for inspection"
    )
    parser.add_argument(
        '--num_printed_translation',
        type=int,
        default=3,
        help="No. of backtranslationed sentences to be printed. "
        "This number will be capped by the batch size. "
        "Sentences will be randomly sampled from the *current* batch"
    )
    parser.add_argument('--val_metric_name', type=str, default="bleu")

    args = parser.parse_args()
    args_dict = vars(args)

    with open(args_dict['config'], 'r') as config_file:
        args_dict.update(yaml.load(config_file, Loader=yaml.SafeLoader))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Good practice: save your training arguments together
    # with the trained model
    if args.models_shared:
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    else:
        torch.save(
            args,
            os.path.join(
                args.output_dir, f"{args.lang1_id}2{args.lang2_id}",
                "training_args.bin"
            )
        )
        torch.save(
            args,
            os.path.join(
                args.output_dir, f"{args.lang2_id}2{args.lang1_id}",
                "training_args.bin"
            )
        )

    # set random seed
    set_seed(args)

    # Start the clock for the beginning of the main function
    start_time = time.time()
    logging.info('Entering main run script')

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    tokenizer = MBartTokenizer.from_pretrained(args.model_path)
    lang1_vocab_constraint = vocab_constraint_from_file(
        tokenizer=tokenizer,
        file=args.lang1_vocab_constrain_file,
        threshold=args.threshold,
        mode='tensor'
    )
    lang1_vocab_constraint = lang1_vocab_constraint.to(device)
    logger.info(
        f"Total valid {args.lang1_id} tokens: {torch.sum(torch.isfinite(lang1_vocab_constraint))}"
    )

    lang2_vocab_constraint = vocab_constraint_from_file(
        tokenizer=tokenizer,
        file=args.lang2_vocab_constrain_file,
        threshold=args.threshold,
        mode='tensor'
    )
    lang2_vocab_constraint = lang2_vocab_constraint.to(device)
    logger.info(
        f"Total valid {args.lang2_id} tokens: {torch.sum(torch.isfinite(lang2_vocab_constraint))}"
    )

    lang1_to_lang2_model = MBartForConditionalGeneration.from_pretrained(
        args.model_path
    )
    lang1_to_lang2_model.to(args.device)
    if args.models_shared:
        lang2_to_lang1_model = lang1_to_lang2_model
    else:
        raise NotImplementedError("Not yet")

    assert args.lang1_id in LANG_ID_2_LANGUAGE_CODES and args.lang2_id in LANG_ID_2_LANGUAGE_CODES
    args.lang1_code = LANG_ID_2_LANGUAGE_CODES[args.lang1_id]
    args.lang2_code = LANG_ID_2_LANGUAGE_CODES[args.lang2_id]
    lang1_dataset = load_dataset(
        "text", data_files=os.path.join(args.data_dir, args.lang1_data_file)
    )["train"]
    lang2_dataset = load_dataset(
        "text", data_files=os.path.join(args.data_dir, args.lang2_data_file)
    )["train"]
    lang1_dataloader = DataLoader(
        lang1_dataset, batch_size=args.batch_size, shuffle=True
    )
    lang2_dataloader = DataLoader(
        lang2_dataset, batch_size=args.batch_size, shuffle=True
    )
    lang1_iter = iter(lang1_dataloader)
    lang2_iter = iter(lang2_dataloader)

    lang2_to_lang1_optimizer = torch.optim.Adam(
        lang2_to_lang1_model.parameters(), lr=args.lr
    )
    if args.models_shared:
        lang1_to_lang2_optimizer = lang2_to_lang1_optimizer
    else:
        lang1_to_lang2_optimizer = torch.optim.Adam(
            lang1_to_lang2_model.parameters(), lr=args.lr
        )

    # lang1_mask = vocab_mask_from_file(tokenizer=tokenizer, file=args.lang1_vocab_constrain_file)
    # lang2_mask = vocab_mask_from_file(tokenizer=tokenizer, file=args.lang2_vocab_constrain_file)

    lang1_meta = LangMeta(
        lang_id=args.lang1_id,
        lang_code=args.lang1_code,
        max_length=args.lang1_max_len
    )

    lang2_meta = LangMeta(
        lang_id=args.lang2_id,
        lang_code=args.lang2_code,
        max_length=args.lang2_max_len
    )
    args.lang1_meta = lang1_meta
    args.lang2_meta = lang2_meta

    backtranslation_pack = BackTranslationPack(
        dataloaders=(lang1_dataloader, lang2_dataloader),
        iterators=(lang1_iter, lang2_iter),
        tokenizer=tokenizer,
        models=(lang1_to_lang2_model, lang2_to_lang1_model),
        metas=(lang1_meta, lang2_meta),
        vocab_constraints=(lang1_vocab_constraint, lang2_vocab_constraint),
        optimizers=(lang1_to_lang2_optimizer, lang2_to_lang1_optimizer)
    )

    if args.do_validation:
        args.val_metric = datasets.load_metric(args.val_metric_name)
        args.val_dataset = load_dataset(
            args.val_dataset_script, args.lang_pair, split="validation"
        )

    main(args, backtranslation_pack)
    print()
