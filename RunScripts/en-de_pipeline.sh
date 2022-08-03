#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
BT_INIT_CONFIG=$2
CAPTIONS_CONFIG=$3
EC_CONFIG=$4
BT_SECONDARY_CONFIG=$5

# Do initial (short) backtranslation
echo "$(date +'%Y-%m-%d %H:%M:%S') en-de_pipeline: Starting initial BT"
python -u BackTranslation/backtranslate.py --config Configs/${BT_INIT_CONFIG}.yml
echo "$(date +'%Y-%m-%d %H:%M:%S') en-de_pipeline: Completed initial BT"

# Do caption training
echo "$(date +'%Y-%m-%d %H:%M:%S') en-de_pipeline: Starting caption training"
python -u -m EC_finetune --config Configs/${CAPTIONS_CONFIG}.yml
echo "$(date +'%Y-%m-%d %H:%M:%S') en-de_pipeline: Completed caption training"

# Do EC
echo "$(date +'%Y-%m-%d %H:%M:%S') en-de_pipeline: Starting EC"
python -u -m EC_finetune --config Configs/${EC_CONFIG}.yml
echo "$(date +'%Y-%m-%d %H:%M:%S') en-de_pipeline: Completed EC"

# # Get translation validation scores after EC training
# #
# # This is a check-in step to observe the results of training
# cp ${OUTPUT_DIR}/bt_init/de-en.en.val ${OUTPUT_DIR}
# cp ${OUTPUT_DIR}/bt_init/de-en.de.val ${OUTPUT_DIR}
# echo "$(date +'%Y-%m-%d %H:%M:%S') en-de_pipeline: Calculating en->de translation validation"
# python -u BackTranslation/translate.py --config Configs/en2de_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# echo "$(date +'%Y-%m-%d %H:%M:%S') en-de_pipeline: Calculating de->en translation validation"
# python -u BackTranslation/translate.py --config Configs/de2en_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# en2de="$(./Tools/bleu.sh ${OUTPUT_DIR}/de-en.en.val.de ${OUTPUT_DIR}/de-en.de.val 13a)"
# de2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/de-en.de.val.en ${OUTPUT_DIR}/de-en.en.val 13a)"
# echo "Post EC: en->de validation bleu: $en2de; de->en validation bleu: $de2en"

# Do rest of backtranslation
echo "$(date +'%Y-%m-%d %H:%M:%S') en-de_pipeline: Starting final BT"
python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml --seed_override 2
echo "$(date +'%Y-%m-%d %H:%M:%S') en-de_pipeline: Completed final BT"
