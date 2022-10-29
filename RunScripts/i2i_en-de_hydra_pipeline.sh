#!/bin/bash

# DATA=clipL
DATA=resnet
SEED=1
EX_ABBR=${DATA}
LANG=en-de
# UNROLL=transformer
UNROLL=recurrent
EC_TYPE=i2i

OUTPUT_ROOT_DIR=Output
OUTPUT_BASE_DIR=${LANG}_pipeline_seed${SEED}
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${EC_TYPE}_bt_sec_${EX_ABBR}

BT_INIT_CONFIG=bt_initial
CAPTIONS_CONFIG=i2i_caption
EC_CONFIG=i2i_ec
BT_SECONDARY_CONFIG=bt_secondary

# Do initial (short) backtranslation
INIT_BT_OUT_DIR=bt_init
# python -u BackTranslation/backtranslate.py \
#     +backtranslate=${BT_INIT_CONFIG} \
#     backtranslate/data=${LANG} \
#     backtranslate.train_eval.seed=${SEED} \
#     backtranslate.output_dir=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${INIT_BT_OUT_DIR}/


# Do caption training
caption_distractor=15
recurrent_hidden_aggregation=false
BT_CKPT_CHOICE=last
CAPTION_OUT_DIR=${EC_TYPE}_captions_${EX_ABBR}_${UNROLL}_distractor${caption_distractor}_hiddenAgg-${recurrent_hidden_aggregation}

# python -u -m EC_finetune +ec=${CAPTIONS_CONFIG} \
#     ec/language=${LANG} \
#     ec/data=${DATA} \
#     ec.train_eval.seed=${SEED} \
#     ec.train_eval.num_distractors_train=${caption_distractor} \
#     ec.train_eval.num_distractors_valid=${caption_distractor} \
#     ec.model.image_unroll=${UNROLL} \
#     ec.model.recurrent_hidden_aggregation=${recurrent_hidden_aggregation} \
#     ec.model.model_name=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${INIT_BT_OUT_DIR}/${BT_CKPT_CHOICE} \
#     ec.output_dir=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${CAPTION_OUT_DIR} \

# Do EC
ec_distractor=15
EC_OUT_DIR=${EC_TYPE}_ec_${EX_ABBR}_${UNROLL}_distractor${ec_distractor}_hiddenAgg-${recurrent_hidden_aggregation} 

# python -u -m EC_finetune  +ec=${EC_CONFIG} \
#     ec/language=${LANG} \
#     ec/data=${DATA} \
#     ec.train_eval.seed=${SEED} \
#     ec.train_eval.num_distractors_train=${ec_distractor} \
#     ec.train_eval.num_distractors_valid=${ec_distractor} \
#     ec.model.image_unroll=${UNROLL} \
#     ec.model.model_name=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${CAPTION_OUT_DIR} \
#     ec.output_dir=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${EC_OUT_DIR}   \

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

# Do rest of backtranslation

# python -u BackTranslation/backtranslate.py \
#     +backtranslate=${BT_SECONDARY_CONFIG} \
#     backtranslate/data=${LANG} \
#     backtranslate.train_eval.seed=${SEED} \
#     backtranslate.model_path=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${EC_OUT_DIR}   \
#     backtranslate.output_dir=${OUTPUT_DIR}

# python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml \
#     --seed_override 2 \
#     --model_dir_override ${OUTPUT_ROOT_DIR}/en-de_pipeline/ec_${EX_ABBR} \
#     --output_dir_override ${OUTPUT_DIR}

# Do test
# cp Data/translation_references/de-en.* ${OUTPUT_DIR}
# python -u BackTranslation/translate.py --config Configs/translate/test_en2de_translate.yaml \
#     --output_dir ${OUTPUT_DIR} \
#     --model_path ${OUTPUT_DIR}/best_bleu
# python -u BackTranslation/translate.py --config Configs/translate/test_de2en_translate.yaml \
#     --output_dir ${OUTPUT_DIR} \
#     --model_path ${OUTPUT_DIR}/best_bleu

# cp ${OUTPUT_ROOT_DIR}/en-de_pipeline/translation_results/* ${OUTPUT_DIR}
./Tools/bleu.sh ${OUTPUT_DIR}/de-en.en.test.de ${OUTPUT_DIR}/de-en.de.test 13a
./Tools/bleu.sh ${OUTPUT_DIR}/de-en.de.test.en ${OUTPUT_DIR}/de-en.en.test 13a
