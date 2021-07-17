from transformers import AutoTokenizer
import multiprocessing as mp
from fsplit.filesplit import Filesplit
import argparse
from typing import Dict, Tuple
import os
import glob
from ipdb import set_trace as bp
from collections import Counter
import json
import io


def tokenize_and_count(args, filepath, tokenizer) -> Tuple[Counter, Counter]:
    tokens_ret = Counter()
    ids_ret = Counter()
    # batch = []
    with open(filepath, "r", encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line == "":
                continue
            line = line.strip()
            # while len(batch) < args.batch_size:
            #     batch.append(line.strip())
            #     continue
            tokens = tokenizer.tokenize(line)
            ids = tokenizer.convert_tokens_to_ids(tokens)
            tokens_ret.update(tokens)
            ids_ret.update(ids)

    # if len(batch) == 0:
    #     return ret

    return tokens_ret, ids_ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_process", help="Number of process to be run in parallel. "
                                            "Also equal to number of files to be split into",
                        type=int, default=-1)
    parser.add_argument("--lang", help="Language to be processed",
                        type=str, default="en")
    parser.add_argument("--extension", help="File extension",
                        type=str, default="txt")
    parser.add_argument("--source_dir", help="Source directory that stores langauge files",
                        type=str, default=".")
    parser.add_argument("--target_dir", help="Target directory that stores temporarily splitted files",
                        type=str, default=None)
    parser.add_argument("--batch_size", help="Number of sentence to be tokenize by tokenizer ",
                        type=int, default=1)
    parser.add_argument("--corpus_name", help="Name of the corpus, only used for naming",
                        type=str, default="cc")

    args = parser.parse_args()
    # load tokenizer
    tokenizer_name = "facebook/mbart-large-cc25"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    # create path to source file
    source_filepath = os.path.join(args.source_dir, f"{args.lang}.{args.extension}")
    assert os.path.exists(source_filepath)
    # split files by byte size
    source_file_size_in_byte = os.path.getsize(source_filepath)
    # bp()
    args.n_process = mp.cpu_count() - 1 if args.n_process in [mp.cpu_count(), -1] else args.n_process
    splitted_file_size_in_byte = source_file_size_in_byte // args.n_process
    # initialize target_dir
    # bp()
    target_dir = os.path.join(args.source_dir, f"{args.lang}_split") if args.target_dir is None else args.target_dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # empty the target directory
    # for f in glob.glob(os.path.join(target_dir, "*")):
    #     os.remove(f)

    def split_cb(f, s):
        print("file: {0}, size: {1}".format(f, s))

    fs = Filesplit()
    # bp()
    fs.split(file=source_filepath, split_size=splitted_file_size_in_byte, output_dir=target_dir)
    # bp()
    splitted_files = glob.glob(os.path.join(target_dir, f"*.{args.extension}"))
    args.n_process = len(splitted_files)

    # start parallel processing
    pool = mp.Pool(processes=args.n_process)
    inputs = [(args, file, tokenizer) for file in splitted_files]

    results = pool.starmap(tokenize_and_count, inputs)
    accumulative_ids_counter = Counter()
    accumulative_tokens_counter = Counter()
    for c in results:
        tokens_ret, ids_ret = c
        accumulative_tokens_counter += tokens_ret
        accumulative_ids_counter += ids_ret
    bp()
    json.dump(accumulative_tokens_counter,
              open(f"{args.source_dir}/{args.lang}_{args.corpus_name}_token2count_dict.{tokenizer_name.replace('/', '-')}.json", "w"))
    json.dump(accumulative_ids_counter,
              open(f"{args.source_dir}/{args.lang}_{args.corpus_name}_tokenID2count_dict.{tokenizer_name.replace('/', '-')}.json", "w"))
    print()



