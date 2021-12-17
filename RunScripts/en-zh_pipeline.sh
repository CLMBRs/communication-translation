#!/bin/bash
source activate unmt

python -u BackTranslation/backtranslate.py --config Configs/en-zh_bt_initial.yml
python -u -m EC_finetune --config Configs/en-zh_captions.yml

# Copy the caption-trained model to the running latest copy
mkdir Output/en-zh_pipeline/last
cp -r Output/en-zh_pipeline/captions/* Output/en-zh_pipeline/last

# Initial EC/BT round does not have weight drift loss
python -u -m EC_finetune --config Configs/en-zh_ec.yml

# Get translation validation scores after EC training
python -u BackTranslation/translate.py --config Configs/en2zh_translate.yml
python -u BackTranslation/translate.py --config Configs/zh2en_translate.yml
./Tools/bleu.sh Output/en-zh_pipeline/zh-en.en.val.zh Output/en-zh_pipeline/zh-en.en.val zh
./Tools/bleu.sh Output/en-zh_pipeline/zh-en.zh.val.en Output/en-zh_pipeline/zh-en.zh.val 13a

# Do backtranslation
python -u BackTranslation/backtranslate.py --config Configs/en-zh_bt_secondary.yml --seed_override 2

# Save end-cycle model to checkpoint folder
mkdir Output/en-zh_pipeline/checkpoint_1
cp -r Output/en-zh_pipeline/last/* Output/en-zh_pipeline/checkpoint_1

for i in {2..12}
do
   python -u -m EC_finetune --config Configs/en-zh_ec.yml --seed_override $((i + 1))

   python -u BackTranslation/translate.py --config Configs/en2zh_translate.yml
   python -u BackTranslation/translate.py --config Configs/zh2en_translate.yml
   ./Tools/bleu.sh Output/en-zh_pipeline/zh-en.en.val.zh Output/en-zh_pipeline/zh-en.zh.val zh
   ./Tools/bleu.sh Output/en-zh_pipeline/zh-en.zh.val.en Output/en-zh_pipeline/zh-en.en.val 13a

   python -u BackTranslation/backtranslate.py --config Configs/en-zh_bt_secondary.yml --seed_override $((i + 1))
   
   mkdir Output/en-zh_pipeline/checkpoint_${i}
   cp -r Output/en-zh_pipeline/last/* Output/en-zh_pipeline/checkpoint_${i}
done
