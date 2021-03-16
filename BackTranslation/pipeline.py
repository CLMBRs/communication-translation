import argparse
import torch
import os
from transformers import pipeline, AutoTokenizer
from EC_finetune.agents import CommunicationAgent


def generate_synthetic_dataset(args):
    training_args = torch.load(args.args_path)
    agent = CommunicationAgent(training_args)
    state_dict = torch.load(args.model_path, map_location=None if torch.cuda.is_available()
                                                                   else torch.device('cpu'))
    agent.load_state_dict(state_dict)
    target_dir = f"{args.data_dir}/{args.source_lang_id}_translated_to_{args.target_lang_id}"
    source_dir = f"{args.data_dir}/{args.source_lang_id}"

    agent.eval()
    tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

    translation_task = f"translation_{args.source_lang_id}_to_{args.target_lang_id}"

    translation = pipeline(translation_task, model=agent.model, tokenizer=tokenizer)

    for filename in os.listdir(source_dir):
        print(filename)
        sent = translator("Hugging Face is a technology company based in New York and Paris", max_length=40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converting Huggingface-based model to fairseq mbart')
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--args_path', type=str)
    parser.add_argument('--source_lang_id', type=str, default="en")
    parser.add_argument('--target_lang_id', type=str, default="ja")
    parser.add_argument('--data_dir', type=str, default="./Data/BackTranslate")
    parser.add_argument('--backtranslate_batch_size', type=int, default=2)
    # parser.add_argument('--source_dir', type=str, default="./Data/BackTranslate")

    args = parser.parse_args()
    if not os.path.exists(args.data_dir):
        os.makedirs(args.data_dir)

    generate_synthetic_dataset(args)

    print()
