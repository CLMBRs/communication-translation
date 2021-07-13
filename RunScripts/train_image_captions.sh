#!/bin/sh
source activate unmt

python -u EC_finetune/__main__.py --config Configs/mbart_captions.yml
