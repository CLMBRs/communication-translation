#!/bin/bash
source activate unmt

echo $(which python)

LANG=$1
DATA=$2
UNROLL=$3
SEED=$4
SEED_OFFSET=$5
if [ ! -z $SEED_OFFSET ]; then
    SEED=$(($SEED + $SEED_OFFSET))
fi

EC_TYPE=t2i
EX_ABBR=${DATA}
export PYTHONPATH=".:${PYTHONPATH}"

OUTPUT_ROOT_DIR=Output
OUTPUT_BASE_DIR=${LANG}/${EC_TYPE}/${DATA}+${UNROLL}/seed${SEED}

CAPTIONS_CONFIG=${EC_TYPE}_caption
EC_CONFIG=${EC_TYPE}_ec
BT_CONFIG=t2i_bt

# Do caption training
BT_CKPT_CHOICE=pretrained
CAPTION_OUT_DIR=captions_from-${BT_CKPT_CHOICE}

python -u -m EC_finetune +ec=${CAPTIONS_CONFIG} \
    ec/language=${LANG} \
    ec/data=${DATA} \
    ec.train_eval.seed=${SEED} \
    ec.model.image_unroll=${UNROLL} \
    ec.output_dir=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${CAPTION_OUT_DIR} \
    ec.model.model_name=facebook/mbart-large-cc25 \

# Do EC
# ec_distractor=15
EC_OUT_DIR=ec_from-${BT_CKPT_CHOICE}

python -u -m EC_finetune  +ec=${EC_CONFIG} \
    ec/language=${LANG} \
    ec/data=${DATA} \
    ec.train_eval.seed=${SEED} \
    ec.model.image_unroll=${UNROLL} \
    ec.model.model_name=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${CAPTION_OUT_DIR} \
    ec.output_dir=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${EC_OUT_DIR}   \


# Do rest of backtranslation
OUTPUT_DIR=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/bt_sec_from-${BT_CKPT_CHOICE}
python -u BackTranslation/backtranslate.py \
    +backtranslate=${BT_CONFIG} \
    backtranslate/data=${LANG} \
    backtranslate.train_eval.seed=$((SEED + 7)) \
    backtranslate.model_path=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${EC_OUT_DIR}   \
    backtranslate.output_dir=${OUTPUT_DIR}

