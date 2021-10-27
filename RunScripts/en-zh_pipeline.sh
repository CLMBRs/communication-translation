#!/bin/bash
source activate unmt

# python -u BackTranslation/backtranslate.py --config Configs/en-zh_bt_initial.yml
python -u EC_finetune/__main__.py --config Configs/en-zh_captions.yml
for i in {1..12}
do
   python -u EC_finetune/__main__.py --config Configs/en-zh_ec.yml --seed_override $((i + 1))
   python -u BackTranslation/backtranslate.py --config Configs/en-zh_bt_secondary.yml --seed_override $((i + 1))
done
