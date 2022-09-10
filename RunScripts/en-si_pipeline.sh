#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
BT_INIT_CONFIG=$2
CAPTIONS_CONFIG=$3
EC_CONFIG=$4
BT_SECONDARY_CONFIG=$5

# Do initial (short) backtranslation
echo "$(date +'%Y-%m-%d %H:%M:%S') en-si_pipeline: Starting initial BT"
python -u BackTranslation/backtranslate.py --config Configs/${BT_INIT_CONFIG}.yml
echo "$(date +'%Y-%m-%d %H:%M:%S') en-si_pipeline: Completed initial BT"

# Do caption training
echo "$(date +'%Y-%m-%d %H:%M:%S') en-si_pipeline: Starting caption training"
python -u -m EC_finetune --config Configs/${CAPTIONS_CONFIG}.yml
echo "$(date +'%Y-%m-%d %H:%M:%S') en-si_pipeline: Completed caption training"

# Do EC
echo "$(date +'%Y-%m-%d %H:%M:%S') en-si_pipeline: Starting EC"
python -u -m EC_finetune --config Configs/${EC_CONFIG}.yml
echo "$(date +'%Y-%m-%d %H:%M:%S') en-si_pipeline: Completed EC"

# # Get translation validation scores after EC training
# #
# # This is a check-in step to observe the results of training
# cp ${OUTPUT_DIR}/bt_init/si-en.en.val ${OUTPUT_DIR}
# cp ${OUTPUT_DIR}/bt_init/si-en.si.val ${OUTPUT_DIR}
# echo "$(date +'%Y-%m-%d %H:%M:%S') en-si_pipeline: Calculating en->si translation validation"
# python -u BackTranslation/translate.py --config Configs/en2si_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# echo "$(date +'%Y-%m-%d %H:%M:%S') en-si_pipeline: Calculating si->en translation validation"
# python -u BackTranslation/translate.py --config Configs/si2en_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# en2si="$(./Tools/bleu.sh ${OUTPUT_DIR}/si-en.en.val.si ${OUTPUT_DIR}/si-en.dsi.val 13a)"
# si2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/si-en.si.val.en ${OUTPUT_DIR}/si-en.en.val 13a)"
# echo "Post EC: en->si validation bleu: $en2si; si->en validation bleu: $si2en"

# Do rest of backtranslation
echo "$(date +'%Y-%m-%d %H:%M:%S') en-si_pipeline: Starting final BT"
python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml --seed_override 2
echo "$(date +'%Y-%m-%d %H:%M:%S') en-si_pipeline: Completed final BT"
