#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
MODEL_CONFIG=$2

# Example default run: test_en-de_pipeline.sh Output/en-de_pipeline Output/en-de_pipeline/best_bleu

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: Calculating en->de test translation scores"
python -u BackTranslation/translate.py --config Configs/test_en2de_translate.yml
en2de="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/de-en.en.test.de Data/translation_references/de-en.de.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: en->de test bleu: $en2de"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: Calculating de->en test translation scores"
python -u BackTranslation/translate.py --config Configs/test_de2en_translate.yml
de2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/de-en.de.test.en Data/translation_references/de-en.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: de->en test bleu: $de2en"
