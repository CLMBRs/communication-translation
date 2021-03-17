import argparse
import torch
import os
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer
from EC_finetune.agents import CommunicationAgent
from BackTranslation.dataloader import MbartMonolingualDataset
from BackTranslation.constant import LANG_ID_2_LANGUAGE_CODES


def generate_synthetic_dataset(args, model, tokenizer):

    target_dir = f"{args.data_dir}/{args.lang1_id}_to_{args.lang2_id}"
    source_dir = f"{args.data_dir}/{args.lang1_id}"
    assert os.path.exists(source_dir)
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    translation_task = f"translation_{args.lang1_id}_to_{args.lang2_id}"

    translation = pipeline(translation_task, model=model, tokenizer=tokenizer)

    for filename in os.listdir(source_dir):
        # print(filename)
        # sent = translation("Hugging Face is a technology company based in New York and Paris", max_length=40)
        if ".txt" not in filename and "head" not in filename:
            continue

        filename_without_suffix = filename.split(".")[0]
        assert "_" in filename_without_suffix
        file_idx = int(filename_without_suffix.split("_")[1])

        with open(f"{source_dir}/{filename}", 'r') as source_file:
            with open(f"{target_dir}/{args.lang2_id}_{file_idx}.txt", "w") as target_file:
                batch_inputs, max_len = [], 0
                for line in tqdm(source_file, desc=f"Translating {filename}"):
                    line = line.strip()
                    if line == "":
                        continue
                    # batch the string input
                    # TODO (Leo): this max_len operation is very language-specific
                    if len(batch_inputs) < args.backtranslate_batch_size:
                        max_len = max(max_len, len(line))
                        batch_inputs.append(line)
                        continue

                    translation_outputs = translation(batch_inputs, max_length=max_len, return_tensors=True)
                    outputs = [d["translation_text"] for d in translation_outputs]
                    # write translated output to the target file
                    for input, output in zip(batch_inputs, outputs):
                        target_file.write(input)
                        target_file.write("\t")
                        target_file.write(output)
                        target_file.write("\n")
                    batch_inputs, max_len = [], 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converting Huggingface-based model to fairseq mbart')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--args_path', type=str)
    parser.add_argument('--lang1_id', type=str, default="ja")
    parser.add_argument('--lang2_id', type=str, default="en")
    parser.add_argument('--data_dir', type=str, default="./Data/BackTranslate")
    parser.add_argument('--backtranslate_batch_size', type=int, default=8)
    # parser.add_argument('--source_dir', type=str, default="./Data/BackTranslate")

    args = parser.parse_args()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

    training_args = torch.load(args.args_path)
    source_to_target_model = CommunicationAgent(training_args)
    state_dict = torch.load(args.model_path, map_location=None if torch.cuda.is_available()
    else torch.device('cpu'))
    source_to_target_model.load_state_dict(state_dict)

    target_to_source_model = CommunicationAgent(training_args)
    state_dict = torch.load(args.model_path, map_location=None if torch.cuda.is_available()
    else torch.device('cpu'))
    target_to_source_model.load_state_dict(state_dict)

    lang1_dataset = MbartMonolingualDataset("/Users/leoliu/My/proj/communication-translation/Data/BackTranslate/ja/ja_1_head100.txt",
                                            tokenizer, LANG_ID_2_LANGUAGE_CODES[args.lang1_id])
    lang2_dataset = MbartMonolingualDataset(
        "/Users/leoliu/My/proj/communication-translation/Data/BackTranslate/ja/ja_1_head100.txt",
        tokenizer, LANG_ID_2_LANGUAGE_CODES[args.lang2_id])

    generate_synthetic_dataset(args, source_to_target_model.model, target_to_source_model.model, tokenizer)

    print()
