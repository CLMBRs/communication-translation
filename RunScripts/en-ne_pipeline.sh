#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
BT_INIT_CONFIG=$2
CAPTIONS_CONFIG=$3
EC_CONFIG=$4
BT_SECONDARY_CONFIG=$5

# Do initial (short) backtranslation
echo "$(date +'%Y-%m-%d %H:%M:%S') en-ne_pipeline: Starting initial BT"
python -u BackTranslation/backtranslate.py --config Configs/${BT_INIT_CONFIG}.yml
echo "$(date +'%Y-%m-%d %H:%M:%S') en-ne_pipeline: Completed initial BT"

# Do caption training
echo "$(date +'%Y-%m-%d %H:%M:%S') en-ne_pipeline: Starting caption training"
python -u -m EC_finetune --config Configs/${CAPTIONS_CONFIG}.yml
echo "$(date +'%Y-%m-%d %H:%M:%S') en-ne_pipeline: Completed caption training"

# Do EC
echo "$(date +'%Y-%m-%d %H:%M:%S') en-ne_pipeline: Starting EC"
python -u -m EC_finetune --config Configs/${EC_CONFIG}.yml
echo "$(date +'%Y-%m-%d %H:%M:%S') en-ne_pipeline: Completed EC"

# # Get translation validation scores after EC training
# #
# # This is a check-in step to observe the results of training
# cp ${OUTPUT_DIR}/bt_init/ne-en.en.val ${OUTPUT_DIR}
# cp ${OUTPUT_DIR}/bt_init/ne-en.ne.val ${OUTPUT_DIR}
# echo "$(date +'%Y-%m-%d %H:%M:%S') en-ne_pipeline: Calculating en->ne translation validation"
# python -u BackTranslation/translate.py --config Configs/en2ne_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# echo "$(date +'%Y-%m-%d %H:%M:%S') en-ne_pipeline: Calculating ne->en translation validation"
# python -u BackTranslation/translate.py --config Configs/ne2en_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# en2ne="$(./Tools/bleu.sh ${OUTPUT_DIR}/ne-en.en.val.ne ${OUTPUT_DIR}/ne-en.dne.val 13a)"
# ne2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/ne-en.ne.val.en ${OUTPUT_DIR}/ne-en.en.val 13a)"
# echo "Post EC: en->ne validation bleu: $en2ne; ne->en validation bleu: $ne2en"

# Do rest of backtranslation
echo "$(date +'%Y-%m-%d %H:%M:%S') en-ne_pipeline: Starting final BT"
python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml --seed_override 2
echo "$(date +'%Y-%m-%d %H:%M:%S') en-ne_pipeline: Completed final BT"
