
import argparse
import torch
import os
from typing import List
from tqdm import tqdm
from datasets import load_metric
from transformers import pipeline, AutoTokenizer
from EC_finetune.agents import CommunicationAgent
from BackTranslation.dataloader import MbartMonolingualDataset
from datasets import load_metric
metric = load_metric("bleu")


def measure_bleu_score(args, model, tokenizer, data_source):
    translation_task = f"translation_{args.data.source_lang_id}_to_{args.data.data.target_lang_id}"
    translation = pipeline(translation_task, model=model, tokenizer=tokenizer)
    # TODO (Leo): I think this function should take a Dataset that already have parallel corpus processed

    metric = load_metric("bleu")
    with open(data_source, "r") as parallel_corpus:
        batch_inputs, max_len = [], 0
        predictions: List[str] = []
        references: List[List[str]] = []
        for line in tqdm(parallel_corpus, desc=f"Translating on data_source"):

            line = line.strip()
            if line == "":
                continue
            source_sent, target_sent = line.split("\t")
            # batch the string input
            # TODO (Leo): this max_len operation is very language-specific, might be useful to be taken care by a Dataset
            if len(batch_inputs) < args.train_eval.backtranslate_batch_size:
                max_len = max(max_len, len(tokenizer.encode(target_sent)))
                batch_inputs.append(source_sent)
                references.append([target_sent])
                continue

            translation_outputs = translation(batch_inputs, max_length=max_len, return_tensors=True)
            outputs = [d["translation_text"] for d in translation_outputs]
            # write translated output to the target file
            predictions.extend(outputs)

            assert len(predictions) == len(references)
            metric.add_batch(predictions=predictions, references=references)
            predictions = references = []
            batch_inputs, max_len = [], 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Measure bleu on parallel corpus')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--args_path', type=str)
    parser.add_argument('--source_lang_id', type=str, default="ja")
    parser.add_argument('--target_lang_id', type=str, default="en")
    parser.add_argument('--data_dir', type=str, default="./Data/BackTranslate")
    parser.add_argument('--backtranslate_batch_size', type=int, default=8)