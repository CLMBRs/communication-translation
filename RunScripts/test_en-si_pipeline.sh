#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
MODEL_CONFIG=$2

# Example default run: test_en-si_pipeline.sh Output/en-si_pipeline Output/en-si_pipeline/best_bleu

# Move golden translation files to the local folder for ease of use
cp -vn DataLink/translation_references/sien.si.test ${OUTPUT_DIR}
cp -vn DataLink/translation_references/sien.en.test ${OUTPUT_DIR}

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-si_pipeline: Calculating en->si test translation scores"
python -u BackTranslation/translate.py \
    --config Configs/test_en2si_translate.yml \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_CONFIG}
en2si="$(./Tools/bleu.sh ${OUTPUT_DIR}/sien.en.test.si ${OUTPUT_DIR}/sien.si.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-si_pipeline: en->si test bleu: $en2si"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-si_pipeline: Calculating si->en test translation scores"
python -u BackTranslation/translate.py \
    --config Configs/test_si2en_translate.yml \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_CONFIG}
si2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/sien.si.test.en ${OUTPUT_DIR}/sien.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-si_pipeline: si->en test bleu: $si2en"
