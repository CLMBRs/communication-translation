#!/bin/sh
# source activate unmt

MODEL='resnet'
SPLIT='val'

python generate_feats.py --data_dir ${SPLIT}2014 \
    --model $MODEL \
    --output_name ${MODEL}_${SPLIT}_feats.pt \
    --output_id_name ${MODEL}_${SPLIT}_feats_id.txt \
    --batch_size 1024 \
    --do_transform \