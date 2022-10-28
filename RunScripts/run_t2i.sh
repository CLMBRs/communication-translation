#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
CAPTIONS_CONFIG=$2
EC_CONFIG=$3
BT_CONFIG=$4

# Do caption training
python -u -m EC_finetune --config Configs/t2i/${CAPTIONS_CONFIG}.yaml

# Do EC
python -u -m EC_finetune --config Configs/t2i/${EC_CONFIG}.yaml

# Do BT
python -u BackTranslation/backtranslate.py --config Configs/t2i/${BT_CONFIG}.yaml
