"""
choose *two* langs from args.langs and use encoder-decoder generate `unconditional` translation 
from source_lang to target_lang (see our definition of `unconditional` below)

This script uses mbart encoder-decoder. For the encoder, we feed empty string ("") + <source_lang_token>; 
since no string from source language is generated, thus "unconditional". For decoder, we prompt the model with <target_lang_token>

Usage:
python unconditional_generate.py <params of your choices>
"""

from EC_finetune.modelings.modeling_mbart import MBartForConditionalGeneration
from transformers import AutoTokenizer
import argparse
import itertools

def main(args):
    generator = MBartForConditionalGeneration.from_pretrained(args.model_path)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    out = []
    for src_lang, tgt_lang in itertools.product(*[args.langs, args.langs]):
        print(f"{src_lang} --> {tgt_lang}")
        tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        # prepare for seq2seq generation
        batch = tokenizer.prepare_seq2seq_batch(src_texts=[""], src_lang=src_lang, return_tensors="pt", max_length=10)
        # generate tokens (idxs); feel free to change kwargs to trigger different decoding algo
        generated_tokens = generator.generate(**batch, decoder_start_token_id=tgt_lang_id, max_length=args.max_gen_length, num_beams=2)
        # get generated tokens in raw string
        generated_string = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        out.append((src_lang, tgt_lang, generated_string))
    # save the generated string to file
    with open(f"{args.model_path}/{','.join(args.langs)}_unconditional_gen_maxlen{args.max_gen_length}.txt", "w") as f:
        for src_lang, tgt_lang, generated_string in out:
            f.write(f"{src_lang} --> {tgt_lang}: {generated_string}\n")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", 
        default="en_XX,zh_CN", 
        help="These are the set of languages (split by comma(,)) we have <source, target> language pairs from"
    ) # ne_NP si_LK zh_CN de_DE ro_RO
    parser.add_argument("--model-path", 
        default="", 
        help="path to the saved model, loadable by `from_pretrained`"
    )
    parser.add_argument("--max-gen-length", type=int, default=50, help="Max leng for generation")  # different lang pair might needs different max length
    args = parser.parse_args()
    args.langs = args.langs.split(',')
    main(args)
