#!/bin/sh
source activate unmt
CONFIG_FILE=$1

python -u EC_finetune/__main__.py --config Configs/$CONFIG_FILE