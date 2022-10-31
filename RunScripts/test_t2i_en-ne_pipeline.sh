#!/bin/bash
source activate unmt

OUTPUT_DIR=$1

# Example default run: test_t2i_en-ne_pipeline.sh Output/en-ne_tec

echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-ne_pipeline: Calculating en->ne test translation scores"
python -u BackTranslation/translate.py --config Configs/t2i/test_en2ne_translate.yaml
en2ne="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/ne-en.en.test.ne Data/translation_references/ne-en.ne.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-ne_pipeline: en->ne test bleu: $en2ne"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-ne_pipeline: Calculating ne->en test translation scores"
python -u BackTranslation/translate.py --config Configs/t2i/test_ne2en_translate.yaml
ne2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/ne-en.ne.test.en Data/translation_references/ne-en.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-ne_pipeline: ne->en test bleu: $ne2en"