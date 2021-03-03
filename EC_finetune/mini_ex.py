# Mask filling only works for bart-large
from transformers import BartTokenizer
from EC_finetune.modelings import BartForConditionalGeneration
tokenizer = BartTokenizer.from_pretrained('facebook/bart-large')
TXT = "My friends are <mask> but they eat too many carbs."
model = BartForConditionalGeneration.from_pretrained('facebook/bart-large')
input_ids = tokenizer([TXT], return_tensors='pt')['input_ids']

logits = model(input_ids).logits
masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
probs = logits[0, masked_index].softmax(dim=0)
values, predictions = probs.topk(5)
a = tokenizer.decode(predictions).split()
print(a)
# ['good', 'great', 'all', 'really', 'very']
