#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
MODEL_CONFIG=$2

# Example default run: test_en-zh_pipeline.sh Output/en-zh_pipeline Output/en-zh_pipeline/best_bleu

# Move golden translation files to the local folder for ease of use
cp -vn DataLink/translation_references/zh-en.zh.test ${OUTPUT_DIR}
cp -vn DataLink/translation_references/zh-en.en.test ${OUTPUT_DIR}

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: Calculating en->zh test translation scores"
python -u BackTranslation/translate.py \
    --config Configs/test_en2zh_translate.yml \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_CONFIG}
en2zh="$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.en.test.zh ${OUTPUT_DIR}/zh-en.zh.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: en->zh test bleu: $en2zh"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: Calculating zh->en test translation scores"
python -u BackTranslation/translate.py \
    --config Configs/test_zh2en_translate.yml \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${MODEL_CONFIG}
zh2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/zh-en.zh.test.en ${OUTPUT_DIR}/zh-en.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: zh->en test bleu: $zh2en"
