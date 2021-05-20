from transformers import AutoTokenizer
import multiprocessing as mp
from fsplit.filesplit import Filesplit
import argparse
from typing import Dict
import os
import glob
from collections import Counter
import json


def tokenize_and_count(args, filepath, tokenizer) -> Counter:
    ret = Counter()
    # batch = []
    with open(filepath, "r") as f:
        for line in f:
            if line == "":
                continue
            line = line.strip()
            # while len(batch) < args.batch_size:
            #     batch.append(line.strip())
            #     continue
            tokens = tokenizer.tokenize(line)
            ret.update(tokens)

    # if len(batch) == 0:
    #     return ret

    return ret


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_process", help="Number of process to be run in parallel. "
                                            "Also equal to number of files to be split into",
                        type=int, default=2)
    parser.add_argument("--lang", help="Language to be processed",
                        type=str, default="zu")
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
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")
    # create path to source file
    source_filepath = os.path.join(args.source_dir, f"{args.lang}.{args.extension}")
    assert os.path.exists(source_filepath)
    # split files by byte size
    source_file_size_in_byte = os.path.getsize(source_filepath)
    args.n_process = mp.cpu_count() - 1 if args.n_process in [mp.cpu_count() or -1] else args.n_process
    splitted_file_size_in_byte = source_file_size_in_byte // args.n_process
    # initialize target_dir
    target_dir = os.path.join(args.source_dir, f"{args.lang}_split") if args.target_dir is None else args.target_dir
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    # empty the target directory
    for f in glob.glob(os.path.join(target_dir, "*")):
        os.remove(f)

    def split_cb(f, s):
        print("file: {0}, size: {1}".format(f, s))

    fs = Filesplit()
    fs.split(file=source_filepath, split_size=splitted_file_size_in_byte, output_dir=target_dir)
    splitted_files = glob.glob(os.path.join(target_dir, f"*.{args.extension}"))
    args.n_process = len(splitted_files)

    # start parallel processing
    pool = mp.Pool(processes=args.n_process)
    inputs = [(args, file, tokenizer) for file in splitted_files]

    results = pool.starmap(tokenize_and_count, inputs)
    accumulative_counter = Counter()
    for c in results:
        accumulative_counter += c
    json.dump(accumulative_counter, open(f"{args.lang}_{args.corpus_name}_count_dict.json", "w"))
    print()



