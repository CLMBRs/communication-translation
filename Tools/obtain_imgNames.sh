#!/bin/sh
# Tools for getting images names txt file directly from *_plus.jsonl file. Basically a one time thing.
python obtain_imgNames_fromJson.py ../Data/captioning_new_new en_captions_train_plus.jsonl train_image_names.txt
python obtain_imgNames_fromJson.py ../Data/captioning_new_new en_captions_val_plus.jsonl val_image_names.txt
python obtain_imgNames_fromJson.py ../Data/ec_finetuning_new en_captions_train_plus.jsonl train_image_names.txt
python obtain_imgNames_fromJson.py ../Data/ec_finetuning_new en_captions_val_plus.jsonl val_image_names.txt