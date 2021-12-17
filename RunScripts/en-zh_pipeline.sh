#!/bin/bash
source activate unmt

python -u BackTranslation/backtranslate.py --config Configs/en-zh_bt_initial.yml
python -u -m EC_finetune --config Configs/en-zh_captions.yml
mkdir Output/en-zh_pipeline/last
cp -r Output/en-zh_pipeline/captions/* Output/en-zh_pipeline/last
# Initial EC/BT round does not have weight drift loss
python -u -m EC_finetune --config Configs/en-zh_ec.yml
python -u BackTranslation/backtranslate.py --config Configs/en-zh_bt_secondary.yml --seed_override 2
mkdir Output/en-zh_pipeline/checkpoint_1
cp -r Output/en-zh_pipeline/last/* Output/en-zh_pipeline/checkpoint_1
for i in {2..12}
do
   python -u -m EC_finetune --config Configs/en-zh_ec.yml --seed_override $((i + 1))
   python -u BackTranslation/backtranslate.py --config Configs/en-zh_bt_secondary.yml --seed_override $((i + 1))
   mkdir Output/en-zh_pipeline/checkpoint_${i}
   cp -r Output/en-zh_pipeline/last/* Output/en-zh_pipeline/checkpoint_${i}
done
