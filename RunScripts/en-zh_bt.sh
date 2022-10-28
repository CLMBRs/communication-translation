#!/bin/bash

DATA=resnet
SEED=$1
EX_ABBR=${DATA}
LANG=en-zh
UNROLL=recurrent

OUTPUT_BASE_DIR=${LANG}_pipeline_seed${SEED}
OUTPUT_DIR=Output/${OUTPUT_BASE_DIR}/bt_sec_${EX_ABBR}
BT_INIT_CONFIG=bt_initial
CAPTIONS_CONFIG=captions
EC_CONFIG=ec
BT_SECONDARY_CONFIG=bt_secondary

# Do initial (short) backtranslation
python -u BackTranslation/backtranslate.py \
    +backtranslate=${BT_INIT_CONFIG} \
    backtranslate/data=${LANG} \
    backtranslate.train_eval.seed=${SEED} \
    backtranslate.output_dir=Output/${OUTPUT_BASE_DIR}/bt_init/

