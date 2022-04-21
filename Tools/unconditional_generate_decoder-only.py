"""
choose *one* langs from args.langs and use decoder-only to generate unconditonal
text target_lang (see our definition of `unconditional` below)

This script uses mbart decoder. We prompt the model with <lang_token> only. 
Since no tokens from the language is provided, we want to know what the model learns about
the generative distribution of the chosen language. thus "unconditional". 

Usage:
python unconditional_generate_decoder-only.py <params of your choices>
"""

from EC_finetune.modelings.modeling_mbart import MBartForCausalLanguageModeling
from transformers import AutoTokenizer
import torch
import argparse


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    vocab = tokenizer.get_vocab()
    generator = MBartForCausalLanguageModeling.from_pretrained(args.model_path)
    generator.config.is_encoder_decoder = False

    out = []
    for src_lang in args.langs:
        print(f"{src_lang}:")
        tokenizer.set_src_lang_special_tokens(src_lang)
        # it's not seq2seq. so we need to prepare the input manually
        batch = {"input_ids": torch.LongTensor([[vocab[src_lang]]]), "attention_mask": torch.LongTensor([[1]])}
        # generate tokens (idxs); feel free to change kwargs to trigger different decoding algo
        generated_tokens = generator.generate(**batch, max_length=args.max_gen_length, num_beams=10, do_sample=False, return_all=False)
        # get generated tokens in raw string
        generated_strings = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        for s in generated_strings:
            out.append((src_lang, s))
        out.append(("", ""))  # this is just for printing purpose
    # save the generated string to file
    with open(f"{args.model_path}/{','.join(args.langs)}_unconditional_gen_maxlen{args.max_gen_length}_beamsearch_finalize.txt", "w") as f:
        for src_lang, generated_string in out:
            f.write(f"{src_lang}: {generated_string}\n")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", 
        default="en_XX,zh_CN,ne_NP,si_LK,de_DE,ro_RO", 
        help="These are the set of languages (split by comma(,)) we will use for generation"
    ) # ne_NP si_LK zh_CN de_DE ro_RO
    parser.add_argument("--model-path", 
        default="", 
        help="path to the saved model, loadable by `from_pretrained`"
    )
    parser.add_argument("--max-gen-length", type=int, default=50, help="Max leng for generation") # different lang pair might needs different max length
    args = parser.parse_args()
    args.langs = args.langs.split(',')
    main(args)
