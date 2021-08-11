#!/bin/sh
source activate unmt

python Tools/prepare_coco.py --image_file Data/coco/full_feats/train_feats \
    --caption_file Data/coco/full_labs/en_train_org \
    --new_ec_train_images images_train \
    --new_ec_train_captions en_captions_train.jsonl \
    --captioning_captions_base en_captions \
    --captioning_images_base images \
    --index2tok_dictionary Data/coco/dics/en_i2w
    
