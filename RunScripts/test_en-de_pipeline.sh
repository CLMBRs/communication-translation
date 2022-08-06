#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
MODEL_CONFIG=$2

# Example default run: test_en-de_pipeline.sh Output/en-de_pipeline Output/en-de_pipeline/best_bleu

# Move golden translation files to the local folder for ease of use
cp -vn DataLink/translation_references/de-en.de.test ${OUTPUT_DIR}
cp -vn DataLink/translation_references/de-en.en.test ${OUTPUT_DIR}

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: Calculating en->de test translation scores"
python -u BackTranslation/translate.py \
    --config Configs/test_en2de_translate.yml \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_CONFIG}
en2de="$(./Tools/bleu.sh ${OUTPUT_DIR}/de-en.en.test.de ${OUTPUT_DIR}/de-en.de.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: en->de test bleu: $en2de"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: Calculating de->en test translation scores"
python -u BackTranslation/translate.py \
    --config Configs/test_de2en_translate.yml \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_CONFIG}
de2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/de-en.de.test.en ${OUTPUT_DIR}/de-en.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: de->en test bleu: $de2en"
