#!/bin/bash
source activate unmt_old

OUTPUT_DIR=Output/en-zh_beit_pipeline
BT_INIT_CONFIG=en-zh_bt_initial
CAPTIONS_CONFIG=en-zh_beit_captions
EC_CONFIG=en-zh_beit_ec_debug
BT_SECONDARY_CONFIG=en-zh_bt_secondary

# # Do initial (short) backtranslation
# python -u BackTranslation/backtranslate.py --config Configs/beit/${BT_INIT_CONFIG}.yml

# Do caption training
# python -u -m EC_finetune --config Configs/beit/${CAPTIONS_CONFIG}.yml

# # Do EC
python -u -m EC_finetune --config Configs/beit/${EC_CONFIG}.yml

# cp ${OUTPUT_DIR}/bt_init/zh-en.en.val ${OUTPUT_DIR}
# cp ${OUTPUT_DIR}/bt_init/zh-en.de.val ${OUTPUT_DIR}

# # Get translation validation scores after EC training
# python -u BackTranslation/translate.py --config Configs/en2zh_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# python -u BackTranslation/translate.py --config Configs/zh2en_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# en2x=$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.en.val.de ${OUTPUT_DIR}/zh-en.zh.val 13a)
# x2en=$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.zh.val.en ${OUTPUT_DIR}/zh-en.en.val 13a)
# echo 'en to zh score: '"$en2x"'; zh to en score: '"$x2en"

# # Do rest of backtranslation
# python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml --seed_override 2
