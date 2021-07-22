from transformers import AutoTokenizer
import json

save_every = 10000

def convert_token2count_file(tokenizer, file, new_file):
    import json
    from tqdm import tqdm
    dic = json.load(open(file, "r"))
    ret = {}
    Notfind = set()
    i = 0
    for t, freq in tqdm(dic.items()):
        if i % save_every == 0:
            json.dump(ret, open(new_file, "w"))
            json.dump(list(Notfind), open(new_file + ".Notfind", "w"))
        try:
            token_id = tokenizer.vocab[t]
            assert token_id not in ret
            ret[token_id] = freq
        except:
            Notfind.add(t)
        i += 1
    json.dump(ret, open(new_file, "w"))
    json.dump(list(Notfind), open(new_file + ".Notfind", "w"))
    # return ret


tokenizer = AutoTokenizer.from_pretrained("facebook/mbart-large-cc25")

# file = "en_cc_token2count_dict.json"
# new_file = "en_cc_tokenID2count_dict.cc25.json"

file = "zh-Hans_cc_token2count_dict.json"
new_file = "zh-Hans_cc_tokenID2count_dict.cc25.json"
convert_token2count_file(tokenizer, file, new_file)
