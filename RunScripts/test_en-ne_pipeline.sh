#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
MODEL_CONFIG=$2

# Example default run: test_en-ne_pipeline.sh Output/en-ne_pipeline Output/en-ne_pipeline/best_bleu

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_pipeline: Calculating en->ne test translation scores"
python -u BackTranslation/translate.py --config Configs/test_en2ne_translate.yml
en2ne="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/ne-en.en.test.ne Data/translation_references/ne-en.ne.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_pipeline: en->ne test bleu: $en2ne"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_pipeline: Calculating ne->en test translation scores"
python -u BackTranslation/translate.py --config Configs/test_ne2en_translate.yml
ne2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/ne-en.ne.test.en Data/translation_references/ne-en.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_pipeline: ne->en test bleu: $ne2en"
