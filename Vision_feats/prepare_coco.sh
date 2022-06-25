#!/bin/sh
source activate unmt

SPLIT='val'
MODEL='beit'

python prepare_coco.py --from_where ${SPLIT}2014 \
    --image_feats ./data/${MODEL}_${SPLIT}_feats.pt \
    --caption_file captions_${SPLIT}2014.json \
    --image_file_name ./data/${MODEL}_${SPLIT}_feats_id.txt \
    --new_ec_train_images images \
    --new_ec_train_captions en_captions \
    --captioning_captions_base en_captions \
    --captioning_images_base images \
    --ec_directory Beit_data/ec_finetuning \
    --captioning_directory Beit_data/captioning \