#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
MODEL_CONFIG=$2

# Example default run: test_en-ne_pipeline.sh Output/en-ne_pipeline Output/en-ne_pipeline/best_bleu

# Move golden translation files to the local folder for ease of use
cp -vn DataLink/translation_references/neen.ne.test ${OUTPUT_DIR}
cp -vn DataLink/translation_references/neen.en.test ${OUTPUT_DIR}

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_pipeline: Calculating en->ne test translation scores"
python -u BackTranslation/translate.py \
    --config Configs/test_en2ne_translate.yml \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_CONFIG}
en2ne="$(./Tools/bleu.sh ${OUTPUT_DIR}/neen.en.test.ne ${OUTPUT_DIR}/neen.ne.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_pipeline: en->ne test bleu: $en2ne"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_pipeline: Calculating ne->en test translation scores"
python -u BackTranslation/translate.py \
    --config Configs/test_ne2en_translate.yml \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_CONFIG}
ne2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/neen.ne.test.en ${OUTPUT_DIR}/neen.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_pipeline: ne->en test bleu: $ne2en"
