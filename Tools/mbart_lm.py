import argparse

import torch
import torch.nn.functional as F

from transformers import MBartTokenizer

from EC_finetune.modelings.modeling_mbart import MBartForCausalLanguageModeling

def main():
    parser = argparse.ArgumentParser(
        description='Interactive mbart language model scoring'
    )
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--input_texts', type=str)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    tokenizer = MBartTokenizer.from_pretrained(args.model_path)
    vocab = tokenizer.get_vocab()
    padding_index = vocab['<pad>']

    lines = [line.strip() for line in open(args.input_texts, 'r')]
    batch = tokenizer(
        lines,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    language_model = MBartForCausalLanguageModeling.from_pretrained(
        args.model_path
    ).to(device)
    language_model.eval()

    lm_targets = batch['input_ids']
    lm_output = language_model(**batch).transpose(1, 2)

    for idx, line in enumerate(lines):
        lm_loss = F.cross_entropy(
            lm_output[idx],
            lm_targets[idx],
            ignore_index=padding_index
        )
        print(f"Sentence: <<{line}>>; NLL: {round(lm_loss.item(), 3)}")


if __name__ == '__main__':
    main()
