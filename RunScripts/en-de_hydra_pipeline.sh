#!/bin/bash

# DATA=clipL
DATA=resnet
SEED=1
EX_ABBR=${DATA}
LANG=en-de
# UNROLL=transformer
UNROLL=recurrent

OUTPUT_ROOT_DIR=Output
OUTPUT_BASE_DIR=${LANG}_pipeline_seed${SEED}
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/bt_sec_${EX_ABBR}

BT_INIT_CONFIG=bt_initial
CAPTIONS_CONFIG=captions
EC_CONFIG=ec
BT_SECONDARY_CONFIG=bt_secondary

# Do initial (short) backtranslation
# python -u BackTranslation/backtranslate.py \
#     +backtranslate=${BT_INIT_CONFIG} \
#     backtranslate/data=${LANG} \
#     backtranslate.train_eval.seed=${SEED} \
#     backtranslate.output_dir=Output/${OUTPUT_BASE_DIR}/bt_init/


# Do caption training
caption_lr=4.0e-5
caption_rep_penalty=1.2
distractor=15
recurrent_hidden_aggregation=false
CAPTION_OUT_DIR=captions_${EX_ABBR}_lr${caption_lr}_${UNROLL}_rep${caption_rep_penalty}_distractor${distractor}_hiddenAgg-${recurrent_hidden_aggregation}
python -u -m EC_finetune +ec=${CAPTIONS_CONFIG} \
    ec/language=${LANG} \
    ec/data=${DATA} \
    ec.train_eval.seed=${SEED} \
    ec.train_eval.lr=${caption_lr} \
    ec.train_eval.num_distractors_train=${distractor} \
    ec.train_eval.num_distractors_valid=${distractor} \
    ec.model.image_unroll=${UNROLL} \
    ec.model.recurrent_hidden_aggregation=${recurrent_hidden_aggregation} \
    ec.model.model_name=Output/${OUTPUT_BASE_DIR}/bt_init/last \
    ec.generation.repetition_penalty=${caption_rep_penalty} \
    ec.output_dir=Output/${OUTPUT_BASE_DIR}/${CAPTION_OUT_DIR} \

# Do EC
ec_rep_penalty=1.0
ec_lr=6.0e-6
EC_OUT_DIR=ec_${EX_ABBR}_lr${ec_lr}_${UNROLL}_rep${ec_rep_penalty}_distractor${distractor}_hiddenAgg-${recurrent_hidden_aggregation} 
python -u -m EC_finetune  +ec=${EC_CONFIG} \
    ec/language=${LANG} \
    ec/data=${DATA} \
    ec.train_eval.seed=${SEED} \
    ec.train_eval.num_distractors_train=${distractor} \
    ec.train_eval.num_distractors_valid=${distractor} \
    ec.model.image_unroll=${UNROLL} \
    ec.model.model_name=Output/${OUTPUT_BASE_DIR}/${CAPTION_OUT_DIR} \
    ec.generation.repetition_penalty=${ec_rep_penalty} \
    ec.output_dir=Output/${OUTPUT_BASE_DIR}/${EC_OUT_DIR}   \

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
#     backtranslate.model_path=Output/${OUTPUT_BASE_DIR}/${EC_OUT_DIR}   \
#     backtranslate.output_dir=${OUTPUT_DIR}

# python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml \
#     --seed_override 2 \
#     --model_dir_override Output/en-de_pipeline/ec_${EX_ABBR} \
#     --output_dir_override ${OUTPUT_DIR}

# Do test
# cp /projects/unmt/communication-translation/Data/translation_references/de-en.* ${OUTPUT_DIR}
# python -u BackTranslation/translate.py --config Configs/test_en2de_translate.yml \
#     --output_dir ${OUTPUT_DIR} \
#     --model_path ${OUTPUT_DIR}/best_bleu
# python -u BackTranslation/translate.py --config Configs/test_de2en_translate.yml \
#     --output_dir ${OUTPUT_DIR} \
#     --model_path ${OUTPUT_DIR}/best_bleu

# cp Output/en-de_pipeline/translation_results/* ${OUTPUT_DIR}
# ./Tools/bleu.sh ${OUTPUT_DIR}/de-en.en.test.de ${OUTPUT_DIR}/de-en.de.test 13a
# ./Tools/bleu.sh ${OUTPUT_DIR}/de-en.de.test.en ${OUTPUT_DIR}/de-en.en.test 13a
