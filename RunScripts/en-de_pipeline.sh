#!/bin/bash
source activate unmt

OUTPUT_DIR=temp/Output/en-de_pipeline_debug
BT_INIT_CONFIG=en-de_bt_initial
CAPTIONS_CONFIG=en-de_captions
EC_CONFIG=en-de_ec
BT_SECONDARY_CONFIG=en-de_bt_secondary

# Do initial (short) backtranslation
python -u BackTranslation/backtranslate.py --config Configs/${BT_INIT_CONFIG}.yml

# # Do caption training
# python -u -m EC_finetune --config Configs/${CAPTIONS_CONFIG}.yml

# # Do EC
# python -u -m EC_finetune --config Configs/${EC_CONFIG}.yml

# cp ${OUTPUT_DIR}/bt_init/de-en.en.val ${OUTPUT_DIR}
# cp ${OUTPUT_DIR}/bt_init/de-en.de.val ${OUTPUT_DIR}

# # Get translation validation scores after EC training
# python -u BackTranslation/translate.py --config Configs/en2de_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# python -u BackTranslation/translate.py --config Configs/de2en_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# en2de=$(./Tools/bleu.sh ${OUTPUT_DIR}/de-en.en.val.de ${OUTPUT_DIR}/de-en.de.val 13a)
# de2en=$(./Tools/bleu.sh ${OUTPUT_DIR}/de-en.de.val.en ${OUTPUT_DIR}/de-en.en.val 13a)
# echo 'en to de score: '"$en2de"'; de to en score: '"$de2en"

# # Do rest of backtranslation
# python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml --seed_override 2
