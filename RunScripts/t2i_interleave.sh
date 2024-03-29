#!/bin/bash
source activate unmt

echo $(which python)
git rev-parse HEAD

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
OUTPUT_BASE_DIR=t2i_interleave/${LANG}/${EC_TYPE}/${DATA}+${UNROLL}/seed${SEED}

CAPTIONS_CONFIG=${EC_TYPE}_caption
EC_CONFIG=${EC_TYPE}_ec
BT_CONFIG=t2i_bt

for i in 2 3 4
do 

    BT_CKPT_CHOICE=last
    CAPTION_OUT_DIR=captions_from-${BT_CKPT_CHOICE}_round${i}
    CAPTION_INIT=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/bt_sec_from-${BT_CKPT_CHOICE}_round$((i - 1))/${BT_CKPT_CHOICE}
    if [[ "$i" == "1" ]]; then
        CAPTION_INIT=facebook/mbart-large-cc25
    fi
    echo ${CAPTION_INIT}
    python -u -m EC_finetune +ec=${CAPTIONS_CONFIG} \
        ec/language=${LANG} \
        ec/data=${DATA} \
        ec.train_eval.seed=${SEED} \
        ec.model.image_unroll=${UNROLL} \
        ec.output_dir=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${CAPTION_OUT_DIR} \
        ec.model.model_name=${CAPTION_INIT} \
        ec.train_eval.max_global_step=512 \
        ec.train_eval.valid_every=128 \
        ec.train_eval.print_every=8

    # if [[ "$i" != "1" ]]; then
    #     rm -rf ${CAPTION_INIT}
    # fi

    # Do EC
    # ec_distractor=15
    EC_OUT_DIR=ec_from-${BT_CKPT_CHOICE}_round${i}
    python -u -m EC_finetune  +ec=${EC_CONFIG} \
        ec/language=${LANG} \
        ec/data=${DATA} \
        ec.train_eval.seed=$((SEED * i)) \
        ec.model.image_unroll=${UNROLL} \
        ec.model.model_name=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${CAPTION_OUT_DIR} \
        ec.output_dir=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${EC_OUT_DIR}   \
        ec.train_eval.max_global_step=512 \
        ec.train_eval.valid_every=64 \
        ec.train_eval.print_every=8 
    
    rm -rf ${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${CAPTION_OUT_DIR}

    # Do rest of backtranslation
    OUTPUT_DIR=bt_sec_from-${BT_CKPT_CHOICE}_round${i}
    python -u BackTranslation/backtranslate.py \
        +backtranslate=${BT_CONFIG} \
        backtranslate/data=${LANG} \
        backtranslate.train_eval.seed=$(((SEED + 0)  * i)) \
        backtranslate.model_path=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${EC_OUT_DIR}   \
        backtranslate.output_dir=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${OUTPUT_DIR}  \
        backtranslate.train_eval.num_steps=2048 \
        backtranslate.train_eval.num_warmup_steps=256 \
        backtranslate.train_eval.num_constrained_steps=512 \
        backtranslate.train_eval.eval_every=128 \
        backtranslate.train_eval.translate_every=256 \
        backtranslate.train_eval.print_every=8 
    
    rm -rf ${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${EC_OUT_DIR}
    echo "end of iteration"
    echo
done
