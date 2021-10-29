import argparse

import torch
import torch.nn.functional as F

import numpy as np
from transformers import MBartTokenizer

from EC_finetune.modelings.modeling_mbart import MBartForConditionalGeneration
from EC_finetune.modelings.modeling_bart import _prepare_bart_decoder_inputs

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

    mbart = MBartForConditionalGeneration.from_pretrained(
        args.model_path
    ).to(device)
    mbart.eval()

    language_model = mbart.model.decoder
    embedding = mbart.model.shared
    output_bias = mbart.final_logits_bias

    lm_ids = batch['input_ids']
    decoder_input_ids, decoder_padding_mask, causal_mask = _prepare_bart_decoder_inputs(
        config=mbart.config,
        input_ids=lm_ids,
        causal_mask_dtype=embedding.weight.dtype
    )
    lm_targets = decoder_input_ids[:,1:].to(device)
    decoder_input_ids = decoder_input_ids[:,:-1].to(device)
    decoder_padding_mask = decoder_padding_mask[:,:-1].to(device)
    causal_mask = causal_mask[:-1,:-1].to(device)
    lm_output = language_model(
        input_ids=decoder_input_ids,
        encoder_hidden_states=None,
        encoder_padding_mask=None,
        decoder_padding_mask=decoder_padding_mask,
        decoder_causal_mask=causal_mask
    )
    lm_logits = F.linear(
        lm_output.last_hidden_state,
        embedding.weight,
        bias=output_bias.to(device)
    )
    for idx, line in enumerate(lines):
        lm_loss = F.cross_entropy(
            lm_logits[idx],
            lm_targets[idx],
            ignore_index=padding_index
        )
        print(f"Sentence: <<{line}>>; NLL: {round(lm_loss.item(), 3)}")


if __name__ == '__main__':
    main()
