#!/bin/bash
source activate unmt

MODEL='beit_ft'
DIR='../../vision_feats'

# bash beit_gen.sh ${DIR} 'train' $MODEL
# bash beit_gen.sh ${DIR} 'val' $MODEL

# # split the generated features
# bash prepare_coco.sh 'train' $MODEL
# bash prepare_coco.sh 'val' $MODEL