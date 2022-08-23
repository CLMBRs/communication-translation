#!/bin/sh
conda activate unmt

INPUT_DIR=$1
SPLIT=$2
MODEL=$3
DIR='./data'

python generate_feats.py --raw_image_dir ${INPUT_DIR}/${SPLIT}2014 \
    --img_encode_model $MODEL \
    --output_name ${DIR}/${MODEL}_${SPLIT}_feats.pt \
    --output_id_name ${DIR}/${MODEL}_${SPLIT}_feats_id.txt \
    --batch_size 2048 \
    --do_transform \
    --read_img_folder 