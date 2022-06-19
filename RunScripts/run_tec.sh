#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
CAPTIONS_CONFIG=$2
EC_CONFIG=$3
BT_CONFIG=$4

# Do caption training
# python -u -m EC_finetune --config Configs/${CAPTIONS_CONFIG}.yml

# Do EC
python -u -m EC_finetune --config Configs/${EC_CONFIG}.yml

# Do BT
python -u BackTranslation/backtranslate.py --config Configs/${BT_CONFIG}.yml
