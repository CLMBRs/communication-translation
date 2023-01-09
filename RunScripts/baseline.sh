#!/bin/bash
source activate unmt

echo $(which python)
git rev-parse HEAD

LANG=$1
SEED=$2
SEED_OFFSET=$3

if [ ! -z $SEED_OFFSET ]; then
    SEED=$(($SEED + $SEED_OFFSET))
fi
export PYTHONPATH=".:${PYTHONPATH}"
# EC_TYPE=$1

OUTPUT_ROOT_DIR=Output
OUTPUT_BASE_DIR=${LANG}_baseline

BT_BASELINE_CONFIG=bt_baseline
echo $BT_BASELINE_CONFIG
# Do initial (short) backtranslation
INIT_BT_OUT_DIR=seed${SEED}
python -u BackTranslation/backtranslate.py \
    +backtranslate=${BT_BASELINE_CONFIG} \
    backtranslate/data=${LANG} \
    backtranslate.train_eval.seed=${SEED} \
    backtranslate.output_dir=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${INIT_BT_OUT_DIR}/ \
    backtranslate.model_path=facebook/mbart-large-cc25 \
