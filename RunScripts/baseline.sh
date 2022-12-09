#!/bin/bash
source activate unmt

echo $(which python)

LANG=$1
SEED=$2
export PYTHONPATH=".:${PYTHONPATH}"
# EC_TYPE=$1

OUTPUT_ROOT_DIR=Output
OUTPUT_BASE_DIR=${LANG}_baseline

BT_BASELINE_CONFIG=bt_baseline

# Do initial (short) backtranslation
INIT_BT_OUT_DIR=seed${SEED}
python -u BackTranslation/backtranslate.py \
    +backtranslate=${BT_INIT_CONFIG} \
    backtranslate/data=${LANG} \
    backtranslate.train_eval.seed=${SEED} \
    backtranslate.output_dir=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${INIT_BT_OUT_DIR}/ \
    backtranslate.model_path=facebook/mbart-large-cc25 \
    backtranslate.train_eval.num_steps=10 \