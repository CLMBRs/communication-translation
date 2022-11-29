#!/bin/sh
source activate unmt

python Tools/finetune_lm.py --config Configs/train_lm.yml
