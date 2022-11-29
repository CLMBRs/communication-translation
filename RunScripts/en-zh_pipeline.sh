#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
BT_INIT_CONFIG=$2
CAPTIONS_CONFIG=$3
EC_CONFIG=$4
BT_SECONDARY_CONFIG=$5

# Do initial (short) backtranslation
echo "$(date +'%Y-%m-%d %H:%M:%S') en-zh_pipeline: Starting initial BT"
python -u BackTranslation/backtranslate.py --config Configs/${BT_INIT_CONFIG}.yml
echo "$(date +'%Y-%m-%d %H:%M:%S') en-zh_pipeline: Completed initial BT"

# Do caption training
echo "$(date +'%Y-%m-%d %H:%M:%S') en-zh_pipeline: Starting caption training"
python -u -m EC_finetune --config Configs/${CAPTIONS_CONFIG}.yml
echo "$(date +'%Y-%m-%d %H:%M:%S') en-zh_pipeline: Completed caption training"

# Do EC
echo "$(date +'%Y-%m-%d %H:%M:%S') en-zh_pipeline: Starting EC"
python -u -m EC_finetune --config Configs/${EC_CONFIG}.yml
echo "$(date +'%Y-%m-%d %H:%M:%S') en-zh_pipeline: Completed EC"

# # Get translation validation scores after EC training
# #
# # This is a check-in step to observe the results of training
# cp ${OUTPUT_DIR}/bt_init/zh-en.en.val ${OUTPUT_DIR}
# cp ${OUTPUT_DIR}/bt_init/zh-en.zh.val ${OUTPUT_DIR}
# echo "$(date +'%Y-%m-%d %H:%M:%S') en-zh_pipeline: Calculating en->zh translation validation"
# python -u BackTranslation/translate.py --config Configs/en2zh_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# echo "$(date +'%Y-%m-%d %H:%M:%S') en-zh_pipeline: Calculating zh->en translation validation"
# python -u BackTranslation/translate.py --config Configs/zh2en_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# en2zh="$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.en.val.zh ${OUTPUT_DIR}/zh-en.dzh.val 13a)"
# zh2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.zh.val.en ${OUTPUT_DIR}/zh-en.en.val 13a)"
# echo "Post EC: en->zh validation bleu: $en2zh; zh->en validation bleu: $zh2en"

# Do rest of backtranslation
echo "$(date +'%Y-%m-%d %H:%M:%S') en-zh_pipeline: Starting final BT"
python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml --seed_override 2
echo "$(date +'%Y-%m-%d %H:%M:%S') en-zh_pipeline: Completed final BT"
