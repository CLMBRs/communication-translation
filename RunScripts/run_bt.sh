#!/bin/sh
source activate unmt
CONFIG_FILE=$1

python BackTranslation/backtranslate.py \
    --config Configs/$CONFIG_FILE \
    --threshold 0.99
