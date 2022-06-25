#!/bin/sh
conda activate unmt

MODEL='beit'
SPLIT='train'
DIR='./data'

python generate_feats.py --data_dir ${SPLIT}2014 \
    --model $MODEL \
    --output_name ${DIR}/${MODEL}_${SPLIT}_feats.pt \
    --output_id_name ${DIR}/${MODEL}_${SPLIT}_feats_id.txt \
    --batch_size 2048 \
    --do_transform \
    --do_downsample \