#!/bin/sh
conda activate unmt

#RAW_IMG_DIR=$1
RAW_SPLIT=$1
SPLIT=$2
IMG_NAME_FILE_DIR=$3
MODEL=$4
DOWNSAMPLE=$5
#OUTPUT_DIR='./Data/beit_ft_captioning'

python ./Vision_feats/generate_feats.py --raw_image_dir /projects/unmt/vision_feats/${RAW_SPLIT}2014 \
    --img_name_file ${IMG_NAME_FILE_DIR}/${SPLIT}_image_names.txt \
    --img_encode_model $MODEL \
    --output_name ${IMG_NAME_FILE_DIR}/${MODEL}_${SPLIT}.pt \
    --downsample $DOWNSAMPLE \
    --batch_size  512 \