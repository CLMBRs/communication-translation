#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
BT_INIT_CONFIG=$2
CAPTIONS_CONFIG=$3
EC_CONFIG=$4
BT_SECONDARY_CONFIG=$5

# Do initial (short) backtranslation
python -u BackTranslation/backtranslate.py --config Configs/${BT_INIT_CONFIG}.yml

# Do caption training
python -u -m EC_finetune --config Configs/${CAPTIONS_CONFIG}.yml

# Do EC
python -u -m EC_finetune --config Configs/${EC_CONFIG}.yml

cp ${OUTPUT_DIR}/bt_init/sien.en.val ${OUTPUT_DIR}
cp ${OUTPUT_DIR}/bt_init/sien.ne.val ${OUTPUT_DIR}

# Get translation validation scores after EC training
# python -u BackTranslation/translate.py --config Configs/en2ne_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# python -u BackTranslation/translate.py --config Configs/ne2en_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# en2ne=$(./Tools/bleu.sh ${OUTPUT_DIR}/neen.en.val.ne ${OUTPUT_DIR}/neen.ne.val 13a)
# ne2en=$(./Tools/bleu.sh ${OUTPUT_DIR}/neen.ne.val.en ${OUTPUT_DIR}/neen.en.val 13a)
# echo 'en to ne score: '"$en2ne"'; ne to en score: '"$ne2en"

# Do rest of backtranslation
python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml --seed_override 2
