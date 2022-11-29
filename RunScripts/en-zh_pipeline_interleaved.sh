#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
BT_INIT_CONFIG=$2
CAPTIONS_CONFIG=$3
EC_CONFIG=$4
BT_SECONDARY_CONFIG=$5

# python -u BackTranslation/backtranslate.py --config Configs/${BT_INIT_CONFIG}.yml

# Get translation validation scores after initial backtranslation
# python -u BackTranslation/translate.py --config Configs/en2zh_translate_init.yml
# python -u BackTranslation/translate.py --config Configs/zh2en_translate_init.yml
# en2zh=$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.en.val.zh ${OUTPUT_DIR}/zh-en.zh.val zh)
# zh2en=$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.zh.val.en ${OUTPUT_DIR}/zh-en.en.val 13a)
# echo 'en to zh score: '"$en2zh"'; zh to en score: '"$zh2en"

python -u -m EC_finetune --config Configs/${CAPTIONS_CONFIG}.yml

# Copy the caption-trained model to the running latest copy
mkdir ${OUTPUT_DIR}/best_loss
cp -r ${OUTPUT_DIR}/captions/* ${OUTPUT_DIR}/best_loss

# Get translation validation scores after caption training
# python -u BackTranslation/translate.py --config Configs/en2zh_translate.yml
# python -u BackTranslation/translate.py --config Configs/zh2en_translate.yml
# en2zh=$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.en.val.zh ${OUTPUT_DIR}/zh-en.zh.val zh)
# zh2en=$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.zh.val.en ${OUTPUT_DIR}/zh-en.en.val 13a)
# echo 'en to zh score: '"$en2zh"'; zh to en score: '"$zh2en"

# Initial EC/BT round does not have weight drift loss
python -u -m EC_finetune --config Configs/${EC_CONFIG}.yml

# Get translation validation scores after EC training
python -u BackTranslation/translate.py --config Configs/en2zh_translate.yml
python -u BackTranslation/translate.py --config Configs/zh2en_translate.yml
en2zh=$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.en.val.zh ${OUTPUT_DIR}/zh-en.zh.val zh)
zh2en=$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.zh.val.en ${OUTPUT_DIR}/zh-en.en.val 13a)
echo 'en to zh score: '"$en2zh"'; zh to en score: '"$zh2en"

# Do backtranslation
python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml --seed_override 2

# Save end-cycle model to checkpoint folder
mkdir ${OUTPUT_DIR}/checkpoint_1
cp -r ${OUTPUT_DIR}/best_loss/* ${OUTPUT_DIR}/checkpoint_1

for i in {2..12}
do
    # EC
    python -u -m EC_finetune --config Configs/${EC_CONFIG}.yml --seed_override $((i + 1))

    # Get translation scores between EC and BT
    python -u BackTranslation/translate.py --config Configs/en2zh_translate.yml
    python -u BackTranslation/translate.py --config Configs/zh2en_translate.yml
    en2zh=$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.en.val.zh ${OUTPUT_DIR}/zh-en.zh.val zh)
    zh2en=$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.zh.val.en ${OUTPUT_DIR}/zh-en.en.val 13a)
    echo 'en to zh score: '"$en2zh"'; zh to en score: '"$zh2en"
    
    # BT
    python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml --seed_override $((i + 1))
   
    # Save to checkpoint
    mkdir ${OUTPUT_DIR}/checkpoint_${i}
    cp -r ${OUTPUT_DIR}/best_loss/* ${OUTPUT_DIR}/checkpoint_${i}
done
