#!/bin/sh
source activate unmt

python -u EC_finetune/train_captions.py --config Configs/mbart_captions.yml
