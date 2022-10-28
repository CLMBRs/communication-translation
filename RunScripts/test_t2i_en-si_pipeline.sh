#!/bin/bash
source activate unmt

OUTPUT_DIR=$1

# Example default run: test_t2i_en-si_pipeline.sh Output/en-si_tec

echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-si_pipeline: Calculating en->si test translation scores"
python -u BackTranslation/translate.py --config Configs/t2i/test_en2si_translate.yaml
en2si="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/si-en.en.test.si Data/translation_references/si-en.si.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-si_pipeline: en->si test bleu: $en2si"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-si_pipeline: Calculating si->en test translation scores"
python -u BackTranslation/translate.py --config Configs/t2i/test_si2en_translate.yaml
si2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/si-en.si.test.en Data/translation_references/si-en.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-si_pipeline: si->en test bleu: $si2en"
