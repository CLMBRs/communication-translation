#!/bin/bash
source activate unmt

python -u BackTranslation/backtranslate.py --config Configs/en-zh_bt_initial_2.yml
python -u -m EC_finetune --config Configs/en-zh_captions_2.yml
mkdir Output/en-zh_pipeline_2/last
cp -r Output/en-zh_pipeline_2/captions/* Output/en-zh_pipeline_2/last
# Initial EC/BT round does not have weight drift loss
python -u -m EC_finetune --config Configs/en-zh_ec_2.yml
python -u BackTranslation/backtranslate.py --config Configs/en-zh_bt_secondary_2.yml --seed_override 2
mkdir Output/en-zh_pipeline_2/checkpoint_1
cp -r Output/en-zh_pipeline_2/last/* Output/en-zh_pipeline_2/checkpoint_1
for i in {2..12}
do
   python -u -m EC_finetune --config Configs/en-zh_ec_2.yml --seed_override $((i + 1))
   python -u BackTranslation/backtranslate.py --config Configs/en-zh_bt_secondary_2.yml --seed_override $((i + 1))
   mkdir Output/en-zh_pipeline_2/checkpoint_${i}
   cp -r Output/en-zh_pipeline_2/last/* Output/en-zh_pipeline_2/checkpoint_${i}
done
