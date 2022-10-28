#!/bin/bash
source activate unmt

OUTPUT_DIR=$1

# Example default run: test_t2i_en-de_pipeline.sh Output/en-de_tec

echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-de_pipeline: Calculating en->de test translation scores"
python -u BackTranslation/translate.py --config Configs/t2i/test_en2de_translate.yaml
en2de="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/de-en.en.test.de Data/translation_references/de-en.de.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-de_pipeline: en->de test bleu: $en2de"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-de_pipeline: Calculating de->en test translation scores"
python -u BackTranslation/translate.py --config Configs/t2i/test_de2en_translate.yaml
de2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/de-en.de.test.en Data/translation_references/de-en.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-de_pipeline: de->en test bleu: $de2en"
