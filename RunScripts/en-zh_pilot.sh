#!/bin/bash
source activate unmt

# Do backtranslation
python -u BackTranslation/backtranslate.py --config Configs/en-zh_pilot_bt_1.yml

# Captions
python -u -m EC_finetune --config Configs/en-zh_pilot_captions.yml

# EC
python -u -m EC_finetune --config Configs/en-zh_pilot_ec.yml

# Get translation validation scores after EC training
python -u BackTranslation/translate.py --config Configs/en2zh_translate_2.yml
python -u BackTranslation/translate.py --config Configs/zh2en_translate_2.yml
en2zh=$(./Tools/bleu.sh Output/en-zh_pilot_2/zh-en.en.val.zh Output/en-zh_pilot_2/zh-en.zh.val zh)
zh2en=$(./Tools/bleu.sh Output/en-zh_pilot_2/zh-en.zh.val.en Output/en-zh_pilot_2/zh-en.en.val 13a)
echo 'en to zh score: '"$en2zh"'; zh to en score: '"$zh2en"

# Do backtranslation
python -u BackTranslation/backtranslate.py --config Configs/en-zh_pilot_bt_2.yml
