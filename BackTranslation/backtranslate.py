import argparse
from argparse import Namespace
import logging
import os
import random
import sys
import time
from collections import defaultdict, namedtuple
from math import ceil
from statistics import mean

import sacrebleu
import torch
import transformers
import yaml
import hydra
import numpy as np
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import MBartTokenizer
from omegaconf import DictConfig, OmegaConf, open_dict

from BackTranslation.constant import LANG_ID_2_LANGUAGE_CODES
from BackTranslation.util import translation2string, LangMeta
from EC_finetune.modelings.modeling_mbart import MBartForConditionalGeneration
from EC_finetune.util import vocab_constraint_from_file
from Util.util import create_logger, set_seed, statbar_string

TOKENIZER_MAP = {
    'zh': 'zh',
}

def namespaced_hparams(hparams):
    if type(hparams) is not dict:
        return hparams
    else:
        hparams = {k: namespaced_hparams(v)
            for k, v in hparams.items()}
        return Namespace(**hparams)


def get_next_batch(dataloader, data_iter):
    try:
        data = next(data_iter)
    except StopIteration:
        # StopIteration is thrown if dataset ends
        # reinitialize data loader
        data_iter = iter(dataloader)
        data = next(data_iter)
    return data


def write_validation_splits(args, source_id, target_id):
    reference_dataset = args.train_eval.val_dataset
    dataloader = torch.utils.data.DataLoader(
        reference_dataset, batch_size=args.train_eval.eval_batch_size
    )
    source_lines = []
    target_lines = []
    num_batch = ceil(args.train_eval.validation_set_size / args.train_eval.eval_batch_size)
    for i, batch in enumerate(dataloader):
        if i == num_batch:
            break
        translation_batch = batch["translation"]
        source_lines += [line for line in translation_batch[source_id]]
        target_lines += [line for line in translation_batch[target_id]]
    source_filename = f"{args.data.lang_pair}.{source_id}.val"
    target_filename = f"{args.data.lang_pair}.{target_id}.val"
    source_file = os.path.join(args.output_dir, source_filename)
    target_file = os.path.join(args.output_dir, target_filename)
    with open(source_file, 'w+') as f:
        for line in source_lines:
            print(line, file=f)
    with open(target_file, 'w+') as f:
        for line in target_lines:
            print(line, file=f)


def save_model(args, backtranslation_pack, saved_model_name):
    tokenizer = backtranslation_pack.tokenizer
    source_meta, target_meta = backtranslation_pack.metas
    source2target_model, target2source_model = backtranslation_pack.models
    source_id, _, _ = list(source_meta)
    target_id, _, _ = list(target_meta)

    # Save the general part of the model
    if args.train_eval.models_shared:
        source2target_model.save_pretrained(
            os.path.join(args.output_dir, saved_model_name)
        )
        tokenizer.save_pretrained(
            os.path.join(args.output_dir, saved_model_name)
        )
    else:
        s2t_path = os.path.join(
            args.output_dir, f"{source_id}2{target_id}", saved_model_name
        )
        t2s_path = os.path.join(
            args.output_dir, f"{target_id}2{source_id}", saved_model_name
        )
        source2target_model.save_pretrained(s2t_path)
        tokenizer.save_pretrained(s2t_path)
        target2source_model.save_pretrained(t2s_path)
        tokenizer.save_pretrained(t2s_path)


def get_translation_score(args, model, tokenizer, source_meta, target_meta):
    reference_dataset = args.train_eval.val_dataset
    cumulative_score = 0
    total_translations = 0
    translation_lines = []

    source_id, source_code, source_max_len = list(source_meta)
    target_id, target_code, target_max_len = list(target_meta)

    TOKENIZER_MAP = {
        'zh': 'zh',
    }

    dataloader = torch.utils.data.DataLoader(
        reference_dataset, batch_size=args.train_eval.eval_batch_size
    )
    num_batch = ceil(args.train_eval.validation_set_size / args.train_eval.eval_batch_size)
    num_batch = min(num_batch, len(dataloader))

    for i, batch in enumerate(
        tqdm(
            dataloader,
            desc=f"val-{args.train_eval.val_metric_name}:{source_id}->{target_id}"
        )
    ):
        if i == num_batch:
            break
        translation_batch = batch["translation"]
        source_string_batch = [
            x for x in translation_batch[source_id] if x.strip() != ''
        ]
        empty_source_indices = [
            idx for idx, x in enumerate(translation_batch[source_id]) if x.strip() == ''
        ]
        reference_string_batch = [
            x for x in translation_batch[target_id] if x.strip() != ''
        ]
        empty_reference_indices = [
            idx for idx, x in enumerate(translation_batch[target_id]) if x.strip() == ''
        ]
        assert empty_source_indices == empty_reference_indices
        with torch.no_grad():
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
                num_beams=args.train_eval.num_beams,
                max_length=target_max_len
            )
            translation_str = tokenizer.batch_decode(
                translated_ids, skip_special_tokens=True
            )
        translation_lines += [line for line in translation_str]

        if target_id in TOKENIZER_MAP:
            score = sacrebleu.corpus_bleu(
                translation_str, [reference_string_batch],
                tokenize=TOKENIZER_MAP[target_id]
            ).score
        else:
            score = sacrebleu.corpus_bleu(
                translation_str, [reference_string_batch]
            ).score

        cumulative_score += score * len(reference_string_batch)
        total_translations += len(reference_string_batch)

    return cumulative_score / total_translations, translation_lines


def get_validation_loss(args, model, tokenizer, source_meta, target_meta):
    reference_dataset = args.train_eval.val_dataset
    loss = 0

    source_id, source_code, source_max_len = list(source_meta)
    target_id, target_code, target_max_len = list(target_meta)

    dataloader = torch.utils.data.DataLoader(
        reference_dataset, batch_size=args.train_eval.eval_batch_size
    )
    num_batch = ceil(args.train_eval.validation_set_size / args.train_eval.eval_batch_size)
    num_batch = min(num_batch, len(dataloader))

    for i, batch in enumerate(
        tqdm(
            dataloader,
            desc=f"val-{args.train_eval.val_metric_name}:{source_id}->{target_id}"
        )
    ):
        if i == num_batch:
            break
        translation_batch = batch["translation"]
        source_string_batch = [
            x for x in translation_batch[source_id] if x.strip() != ''
        ]
        reference_string_batch = [
            x for x in translation_batch[target_id] if x.strip() != ''
        ]
        with torch.no_grad():
            parallel_batch = tokenizer.prepare_seq2seq_batch(
                source_string_batch,
                src_lang=source_code,
                tgt_lang=target_code,
                tgt_texts=reference_string_batch,
                max_length=source_max_len,
                max_target_length=target_max_len,
                return_tensors="pt"
            )
            parallel_batch = parallel_batch.to(args.device)
            output = model(**parallel_batch)
            batch_loss = output.loss.item()
            loss += batch_loss / num_batch

    return loss


def evaluate(
    args,
    backtranslation_pack,
    best_score,
    patience_count,
    step,
    logger,
    mode='translate'
):
    if mode not in ['translate', 'crossentropy']:
        raise ValueError(f"Evaluation mode {mode} is not recognized")

    source2target_model, target2source_model = backtranslation_pack.models
    tokenizer = backtranslation_pack.tokenizer
    source_meta, target_meta = backtranslation_pack.metas
    source_id, _, _ = list(source_meta)
    target_id, _, _ = list(target_meta)

    source2target_model.eval()
    target2source_model.eval()

    if mode == 'translate':
        source2target_score, source2target_translations = get_translation_score(
            args, source2target_model, tokenizer, source_meta, target_meta
        )
        target2source_score, target2source_translations = get_translation_score(
            args, target2source_model, tokenizer, target_meta, source_meta
        )
        metric_name = 'bleu'
    else:
        source2target_score = get_validation_loss(
            args, source2target_model, tokenizer, source_meta, target_meta
        )
        target2source_score = get_validation_loss(
            args, target2source_model, tokenizer, target_meta, source_meta
        )
        metric_name = 'loss'

    mean_score = round(mean([source2target_score, target2source_score]), 4)
    stats = {
        'step': step + 1,
        'mode': 'validation',
        f'{target_id} {metric_name}': round(source2target_score, 4),
        f'{source_id} {metric_name}': round(target2source_score, 4),
        f'mean {metric_name}': mean_score
    }
    logger.info(statbar_string(stats))

    if mode == 'translate':
        data_file = os.path.join(args.output_dir, args.data.output_data_filename)
        metrics = [
            step + 1,
            round(source2target_score, 2),
            round(target2source_score, 2)
        ]
        with open(data_file, 'a') as f:
            print(", ".join([str(x) for x in metrics]), file=f)

    new_score_best = (
        (mean_score > best_score) if mode == 'translate' else
        (mean_score < best_score)
    )

    if new_score_best:
        # if we encounter a better model, we restart patience counting
        # and save the model
        patience_count = 0
        best_score = mean_score
        if step > 0:
            logger.info(
                f"New best mean {metric_name} {mean_score}"
                f" at step {step + 1}, saving"
            )
            save_model(
                args, backtranslation_pack, saved_model_name=f"best_{metric_name}"
            )
        if mode == 'translate':
            source_filename = f"{args.data.lang_pair}.{source_id}.val.{target_id}"
            target_filename = f"{args.data.lang_pair}.{target_id}.val.{source_id}"
            source_file = os.path.join(args.output_dir, source_filename)
            target_file = os.path.join(args.output_dir, target_filename)
            with open(source_file, 'w+') as f:
                for line in source2target_translations:
                    print(line, file=f)
            with open(target_file, 'w+') as f:
                for line in target2source_translations:
                    print(line, file=f)
    else:
        if step >= args.train_eval.early_stop_start_time:
            # we start counting the early stopping after some 'warmup period'
            patience_count += 1

    return best_score, patience_count

def backtranslate(args, backtranslation_pack, logger):
    checkpoint_stats = defaultdict(list)
    best_translation_score = 0.0
    best_crossentropy = float('inf')
    translation_patience_count = 0
    crossent_patience_count = 0
    
    if args.train_eval.do_initial_eval:
        best_crossentropy, crossent_patience_count = evaluate(
            args, backtranslation_pack, best_crossentropy,
            crossent_patience_count, 0, logger, mode='crossentropy'
        )
        crossent_patience_count = 0

    for step in range(args.train_eval.num_steps):
        # we might want to randomly decide the order, because we don't want the
        # model to learn the pattern that we do, e.g., English first and then
        # Japanese second.

        if (step + 1) % args.train_eval.print_every == 0 and args.train_eval.print_translation:
            translation_results = {args.data.lang1_id: [], args.data.lang2_id: []}

        mini_step = 0
        for source in random.sample([0, 1], 2):
            target = np.abs(1 - source)

            source_dataloader = backtranslation_pack.dataloaders[source]
            source_data_iter = backtranslation_pack.iterators[source]
            tokenizer = backtranslation_pack.tokenizer
            source2target_model = backtranslation_pack.models[source]
            target2source_model = backtranslation_pack.models[target]
            target_vocab_constraint = (
                backtranslation_pack.vocab_constraints[target]
            )
            target_secondary_constraint = (
                backtranslation_pack.secondary_constraints[target]
            )
            target2source_optimizer = backtranslation_pack.optimizers[target]
            target2source_scheduler = backtranslation_pack.schedulers[target]
            source_meta = backtranslation_pack.metas[source]
            target_meta = backtranslation_pack.metas[target]

            source_id, source_code, source_max_len = list(source_meta)
            target_id, target_code, target_max_len = list(target_meta)

            if step == 0:
                write_validation_splits(args, source_id, target_id)

            # 1. we use source2target_model to generate synthetic text in target
            # language
            source2target_model.eval()
            with torch.no_grad():
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
                if step < args.train_eval.num_constrained_steps:
                    translated_tokens = source2target_model.generate(
                        **source_batch,
                        decoder_start_token_id=tokenizer.
                        lang_code_to_id[target_code],
                        max_length=target_max_len,
                        lang_mask=target_vocab_constraint
                    )
                elif target_secondary_constraint is not None:
                    translated_tokens = source2target_model.generate(
                        **source_batch,
                        decoder_start_token_id=tokenizer.
                        lang_code_to_id[target_code],
                        max_length=target_max_len,
                        lang_mask=target_secondary_constraint
                    )
                else:
                    translated_tokens = source2target_model.generate(
                        **source_batch,
                        decoder_start_token_id=tokenizer.
                        lang_code_to_id[target_code],
                        max_length=target_max_len,
                    )

            # turn the predicted subtokens into sentence in string
            translation = tokenizer.batch_decode(
                translated_tokens, skip_special_tokens=True
            )
            if (step + 1) % args.train_eval.print_every == 0 and args.train_eval.print_translation:
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
            nn.utils.clip_grad_norm_(
                target2source_model.parameters(), args.train_eval.grad_clip
            )
            target2source_optimizer.step()
            # Make sure if the model optimizer/scheuler are shared that
            # we only step once per "main" step rather than once per
            # language
            if (not args.train_eval.models_shared) or (mini_step == 1):
                target2source_scheduler.step()
            mini_step += 1
            checkpoint_stats["loss"].append(
                output["loss"].detach().cpu().item()
            )

        if (step + 1) % args.train_eval.print_every == 0:
            checkpoint_average_stats = {}
            checkpoint_average_stats['step'] = step + 1
            checkpoint_average_stats['mode'] = "train"
            checkpoint_average_stats['lr'] = round(
                target2source_scheduler.get_last_lr()[0], 8
            )
            for key, value in checkpoint_stats.items():
                checkpoint_average_stats[key] = round(np.mean(value), 4)
            logger.info(statbar_string(checkpoint_average_stats))
            checkpoint_stats = defaultdict(list)
            if args.train_eval.print_translation:
                logger.info(
                    translation2string(
                        translation_results, args.train_eval.num_printed_translation
                    )
                )

        if args.train_eval.do_crossent_eval and (step + 1) % args.train_eval.eval_every == 0:
            best_crossentropy, crossent_patience_count = evaluate(
                args, backtranslation_pack, best_crossentropy,
                crossent_patience_count, step, logger, mode='crossentropy'
            )
            if (
                hasattr(args, 'crossent_patience') and
                crossent_patience_count > args.train_eval.crossent_patience
            ):
                break
        if args.train_eval.do_translate_eval and (step + 1) % args.train_eval.translate_every == 0:
            best_translation_score, translation_patience_count = evaluate(
                args, backtranslation_pack, best_translation_score,
                translation_patience_count, step, logger
            )
            if (
                hasattr(args, 'translation_patience') and
                translation_patience_count > args.translation_patience
            ):
                break

    save_model(args, backtranslation_pack, saved_model_name="last")
    logger.info("training complete; final model state saved")


BackTranslationPack = namedtuple(
    "BackTranslationPack", [
        "dataloaders", "iterators", "tokenizer", "models", "metas",
        "vocab_constraints", "secondary_constraints", "optimizers", "schedulers"
    ]
)

@hydra.main(version_base=None, config_path="../Configs")
def main(args: DictConfig) -> None:
    logger = create_logger(name="backtranslate")
    container = OmegaConf.to_object(args)
    # turn hparams into a namespace
    args = namespaced_hparams(container['backtranslate'])

    args.output_dir = os.path.join(args.backtranslated_dir, args.output_dir)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Good practice: save your training arguments together
    # with the trained model
    if args.train_eval.models_shared:
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
    else:
        torch.save(
            args,
            os.path.join(
                args.output_dir, f"{args.data.lang1_id}2{args.data.lang2_id}",
                "training_args.bin"
            )
        )
        torch.save(
            args,
            os.path.join(
                args.output_dir, f"{args.data.lang2_id}2{args.data.lang1_id}",
                "training_args.bin"
            )
        )

    # set random seed
    set_seed(args.train_eval.seed, args.train_eval.n_gpu)

    # Start the clock for the beginning of the main function
    start_time = time.time()
    logging.info('Entering main run script')

    # Setup CUDA, GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    tokenizer = MBartTokenizer.from_pretrained(args.model_path)

    if not hasattr(args.train_eval, 'num_constrained_steps'):
        args.train_eval.num_constrained_steps = 0

    # Write the model description
    logger.info("Configuration:")
    print(args)

    lang1_vocab_constraint = vocab_constraint_from_file(
        tokenizer=tokenizer,
        file=args.data.lang1_vocab_constrain_file,
        threshold=args.train_eval.vocab_constraint_threshold,
        mode='tensor'
    )
    lang1_vocab_constraint = lang1_vocab_constraint.to(device)
    logger.info(
        f"Total valid {args.data.lang1_id} tokens:"
        f" {torch.sum(torch.isfinite(lang1_vocab_constraint))}"
    )

    lang2_vocab_constraint = vocab_constraint_from_file(
        tokenizer=tokenizer,
        file=args.data.lang2_vocab_constrain_file,
        threshold=args.train_eval.vocab_constraint_threshold,
        mode='tensor'
    )
    lang2_vocab_constraint = lang2_vocab_constraint.to(device)
    logger.info(
        f"Total valid {args.data.lang2_id} tokens:"
        f" {torch.sum(torch.isfinite(lang2_vocab_constraint))}"
    )

    if hasattr(args, 'secondary_threshold'):
        lang1_secondary_constraint = vocab_constraint_from_file(
            tokenizer=tokenizer,
            file=args.data.lang1_vocab_constrain_file,
            threshold=args.train_eval.secondary_threshold,
            mode='tensor'
        )
        logger.info(
            f"Total secondary {args.data.lang1_id} tokens:"
            f" {torch.sum(torch.isfinite(lang1_secondary_constraint))}"
        )
        lang1_secondary_constraint = lang1_secondary_constraint.to(device)
        lang2_secondary_constraint = vocab_constraint_from_file(
            tokenizer=tokenizer,
            file=args.data.lang2_vocab_constrain_file,
            threshold=args.train_eval.secondary_threshold,
            mode='tensor'
        )
        logger.info(
            f"Total secondary {args.data.lang2_id} tokens:"
            f" {torch.sum(torch.isfinite(lang2_secondary_constraint))}"
        )
        lang2_secondary_constraint = lang2_secondary_constraint.to(device)
    else:
        lang1_secondary_constraint = lang2_secondary_constraint = None

    lang1_to_lang2_model = MBartForConditionalGeneration.from_pretrained(
        args.model_path
    )
    lang1_to_lang2_model.to(args.device)
    if args.train_eval.models_shared:
        lang2_to_lang1_model = lang1_to_lang2_model
    else:
        raise NotImplementedError("Not yet")

    assert (
        args.data.lang1_id in LANG_ID_2_LANGUAGE_CODES and
        args.data.lang2_id in LANG_ID_2_LANGUAGE_CODES
    )
    args.data.lang1_code = LANG_ID_2_LANGUAGE_CODES[args.data.lang1_id]
    args.data.lang2_code = LANG_ID_2_LANGUAGE_CODES[args.data.lang2_id]
    logger.info(
        f"Source language code: {args.data.lang1_code},"
        f" target language code: {args.data.lang2_code}"
    )

    logger.info(f"Loading {args.data.lang1_code} and {args.data.lang2_code} datasets")
    lang1_dataset = load_dataset(
        "text", data_files=os.path.join(args.data.data_dir, args.data.lang1_data_file)
    )["train"]
    lang2_dataset = load_dataset(
        "text", data_files=os.path.join(args.data.data_dir, args.data.lang2_data_file)
    )["train"]

    logger.info(f"Creating {args.data.lang1_code} and {args.data.lang2_code} dataloaders")
    lang1_dataloader = DataLoader(
        lang1_dataset, batch_size=args.train_eval.batch_size, shuffle=True
    )
    lang2_dataloader = DataLoader(
        lang2_dataset, batch_size=args.train_eval.batch_size, shuffle=True
    )

    lang1_iter = iter(lang1_dataloader)
    lang2_iter = iter(lang2_dataloader)

    lang2_to_lang1_optimizer = torch.optim.Adam(
        lang2_to_lang1_model.parameters(), lr=args.train_eval.lr
    )
    if args.train_eval.models_shared:
        lang1_to_lang2_optimizer = lang2_to_lang1_optimizer
    else:
        lang1_to_lang2_optimizer = torch.optim.Adam(
            lang1_to_lang2_model.parameters(), lr=args.train_eval.lr
        )

    if args.train_eval.schedule == 'linear_w_warmup':
        scheduler_method = transformers.get_linear_schedule_with_warmup
        scheduler_args = {
            'optimizer': lang1_to_lang2_optimizer,
            'num_warmup_steps': args.train_eval.num_warmup_steps,
            'num_training_steps': args.train_eval.num_steps
        }
    else:
        # Default to constant schedule with warmup
        scheduler_method = transformers.get_constant_schedule_with_warmup
        scheduler_args = {
            'optimizer': lang1_to_lang2_optimizer,
            'num_warmup_steps': args.train_eval.num_warmup_steps
        }

    lang1_to_lang2_scheduler = scheduler_method(**scheduler_args)
    if args.train_eval.models_shared:
        lang2_to_lang1_scheduler = lang1_to_lang2_scheduler
    else:
        scheduler_args['optimizer'] = lang2_to_lang1_optimizer
        lang2_to_lang1_scheduler = scheduler_method(**scheduler_args)

    lang1_meta = LangMeta(
        lang_id=args.data.lang1_id,
        lang_code=args.data.lang1_code,
        max_length=args.data.lang1_max_len
    )

    lang2_meta = LangMeta(
        lang_id=args.data.lang2_id,
        lang_code=args.data.lang2_code,
        max_length=args.data.lang2_max_len
    )
    args.data.lang1_meta = lang1_meta
    args.data.lang2_meta = lang2_meta

    backtranslation_pack = BackTranslationPack(
        dataloaders=(lang1_dataloader, lang2_dataloader),
        iterators=(lang1_iter, lang2_iter),
        tokenizer=tokenizer,
        models=(lang1_to_lang2_model, lang2_to_lang1_model),
        metas=(lang1_meta, lang2_meta),
        vocab_constraints=(lang1_vocab_constraint, lang2_vocab_constraint),
        optimizers=(lang1_to_lang2_optimizer, lang2_to_lang1_optimizer),
        schedulers=(lang1_to_lang2_scheduler, lang2_to_lang1_scheduler),
        secondary_constraints=(
            lang1_secondary_constraint, lang2_secondary_constraint
        )
    )

    if args.train_eval.do_crossent_eval or args.train_eval.do_translate_eval:
        args.train_eval.val_dataset = load_dataset(
            args.train_eval.val_dataset_script, args.data.lang_pair, split="validation"
        )
        data_columns = [
            "step", f"{args.data.lang1_id} to {args.data.lang2_id}",
            f"{args.data.lang2_id} to {args.data.lang1_id}"
        ]
        data_file = os.path.join(args.output_dir, args.data.output_data_filename)
        
        with open(data_file, 'w+') as f:
            print(", ".join(data_columns), file=f)

    backtranslate(args, backtranslation_pack, logger)

if __name__ == '__main__':
    main()
