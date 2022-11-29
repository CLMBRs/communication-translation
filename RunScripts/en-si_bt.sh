#!/bin/bash

DATA=clipL
# DATA=resnet
SEED=$1
LANG=en-si
# EC_TYPE=$1

OUTPUT_ROOT_DIR=Output
OUTPUT_BASE_DIR=${LANG}_pipeline_seed${SEED}

BT_INIT_CONFIG=bt_initial
BT_SECONDARY_CONFIG=bt_secondary

# Do initial (short) backtranslation
INIT_BT_OUT_DIR=bt_init
python -u BackTranslation/backtranslate.py \
    +backtranslate=${BT_INIT_CONFIG} \
    backtranslate/data=${LANG} \
    backtranslate.train_eval.seed=${SEED} \
    backtranslate.train_eval.val_dataset_script=BackTranslation/flores/flores.py \
    backtranslate.output_dir=${OUTPUT_ROOT_DIR}/${OUTPUT_BASE_DIR}/${INIT_BT_OUT_DIR}/
