from transformers import AutoTokenizer
import json


def convert_token2count_file(tokenizer, file):
    import json
    from tqdm import tqdm
    dic = json.load(open(file, "r"))
    ret = {}
    for t, freq in tqdm(dic.items()):
        token_id = tokenizer.vocab[t]
        assert token_id not in ret
        ret[token_id] = freq
    return ret


tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-50")

file = "en_cc_token2count_dict.json"
new_dict = convert_token2count_file(tokenizer, file)
json.dump(new_dict, open("en_cc_tokenID2count_dict.json", "w"))

# file = "zh-Hans_cc_token2count_dict.json"
# new_dict = convert_token2count_file(tokenizer, file)
# json.dump(new_dict, open("zh-Hans_cc_tokenID2count_dict.json", "w"))