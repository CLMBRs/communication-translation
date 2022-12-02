#!/bin/bash
source activate unmt

echo $(which python)

LANG=$1
DATA=$2
UNROLL=$3
SEED=$4
EC_TYPE=t2i
EX_ABBR=${DATA}

OUTPUT_ROOT_DIR=Output
OUTPUT_BASE_DIR=${EC_TYPE}_${LANG}_seed${SEED}_all15

CAPTIONS_CONFIG=${EC_TYPE}_caption
EC_CONFIG=${EC_TYPE}_ec
BT_CONFIG=t2i_bt


# Do caption training
# BT_CKPT_CHOICE=last
BT_CKPT_CHOICE=pretrained
caption_distractor=15
CAPTION_OUT_DIR=${EC_TYPE}_captions_${EX_ABBR}_${UNROLL}_from-${BT_CKPT_CHOICE}

python -u -m EC_finetune +ec=${CAPTIONS_CONFIG} \
    ec/language=${LANG} \
    ec/data=${DATA} \
    ec.train_eval.seed=${SEED} \
    ec.train_eval.num_distractors_train=${caption_distractor} \
    ec.train_eval.num_distractors_valid=${caption_distractor} \
    ec.model.image_unroll=${UNROLL} \
    ec.output_dir=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${CAPTION_OUT_DIR} \
    ec.model.model_name=facebook/mbart-large-cc25 \

# Do EC
ec_distractor=15
EC_OUT_DIR=${EC_TYPE}_ec_${EX_ABBR}_${UNROLL}_from-${BT_CKPT_CHOICE}

python -u -m EC_finetune  +ec=${EC_CONFIG} \
    ec/language=${LANG} \
    ec/data=${DATA} \
    ec.train_eval.seed=${SEED} \
    ec.model.image_unroll=${UNROLL} \
    ec.model.model_name=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${CAPTION_OUT_DIR} \
    ec.output_dir=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${EC_OUT_DIR}   \

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


OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${EC_TYPE}_bt_sec_${EX_ABBR}_${UNROLL}_from-${BT_CKPT_CHOICE}
# Do rest of backtranslation

PYTHONPATH=. python -u BackTranslation/backtranslate.py \
    +backtranslate=${BT_CONFIG} \
    backtranslate/data=${LANG} \
    backtranslate.train_eval.seed=$$((SEED + 7)) \
    backtranslate.model_path=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${EC_OUT_DIR}   \
    backtranslate.output_dir=${OUTPUT_DIR}


# Do test
# cp Data/translation_references/de-en.* ${OUTPUT_DIR}
# python -u BackTranslation/translate.py --config Configs/translate/test_en2de_translate.yaml \
#     --output_dir ${OUTPUT_DIR} \
#     --model_path ${OUTPUT_DIR}/best_bleu
# python -u BackTranslation/translate.py --config Configs/translate/test_de2en_translate.yaml \
#     --output_dir ${OUTPUT_DIR} \
#     --model_path ${OUTPUT_DIR}/best_bleu

# cp ${OUTPUT_ROOT_DIR}/en-de_pipeline/translation_results/* ${OUTPUT_DIR}
# ./Tools/bleu.sh ${OUTPUT_DIR}/de-en.en.test.de ${OUTPUT_DIR}/de-en.de.test 13a
# ./Tools/bleu.sh ${OUTPUT_DIR}/de-en.de.test.en ${OUTPUT_DIR}/de-en.en.test 13a