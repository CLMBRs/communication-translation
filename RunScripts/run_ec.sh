#!/bin/bash
source activate unmt
CONFIG_FILE=$1

python -u -m EC_finetune --config Configs/$CONFIG_FILE
