#!/bin/sh
source activate unmt
CONFIG_FILE=$1

python -u BackTranslation/translate.py \
    --config Configs/$CONFIG_FILE
