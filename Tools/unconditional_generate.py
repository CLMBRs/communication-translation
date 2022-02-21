from EC_finetune.modelings.modeling_mbart import MBartForConditionalGeneration
from transformers import AutoTokenizer, MBartTokenizerFast, MBartTokenizer
import argparse
import torch
import itertools
from pdb import set_trace 

def main(args):
    # model_path = "/home1/zliu9986/communication-translation/Models/mbart-large-cc25"
    generator = MBartForConditionalGeneration.from_pretrained(args.model_path)
    # set_trace()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    out = []
    for src_lang, tgt_lang in itertools.product(*[args.langs, args.langs]):
    # for lang in args.langs:
        print(f"{src_lang} --> {tgt_lang}")
        tgt_lang_id = tokenizer.convert_tokens_to_ids(tgt_lang)
        # lang_id = torch.LongTensor([lang_id])
        # tokenizer.src_lang = lang
        # encoded_tokens = tokenizer(lang, return_tensors="pt")
        # set_trace()
        batch = tokenizer.prepare_seq2seq_batch(src_texts=[""], src_lang=src_lang, return_tensors="pt", max_length=10)
        generated_tokens = generator.generate(**batch, decoder_start_token_id=tgt_lang_id, max_length=args.max_gen_length, num_beams=1)
        generated_string = tokenizer.batch_decode(generated_tokTens, skip_special_tokens=True)[0]
        out.append((src_lang, tgt_lang, generated_string))
        set_trace()
        # print(generated_string)
    with open(f"{args.model_path}/{','.join(args.langs)}_unconditional_gen_maxlen{args.max_gen_length}.txt", "w") as f:
        for src_lang, tgt_lang, generated_string in out:
            f.write(f"{src_lang} --> {tgt_lang}: {generated_string}\n")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--langs", default="en_XX,si_LK")
    # parser.add_argument("--model-path", default="/home1/zliu9986/communication-translation/Models/bt_en-de_baseline/last",)
    parser.add_argument("--model-path", default="/home1/zliu9986/communication-translation/Models/bt_en-si_baseline/last",)
    parser.add_argument("--max-gen-length", type=int, default=50,)
    args = parser.parse_args()
    args.langs = args.langs.split(',')
    main(args)