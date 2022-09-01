#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
MODEL_CONFIG=$2

# Example default run: test_en-zh_pipeline.sh Output/en-zh_pipeline Output/en-zh_pipeline/best_bleu

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: Calculating en->zh test translation scores"
python -u BackTranslation/translate.py --config Configs/test_en2zh_translate.yml
# Special note that Chinese tokenization used here (zh) is different from the default setting (13a)
en2zh="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/zh-en.en.test.zh Data/translation_references/zh-en.zh.test zh)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: en->zh test bleu: $en2zh"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: Calculating zh->en test translation scores"
python -u BackTranslation/translate.py --config Configs/test_zh2en_translate.yml
zh2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/zh-en.zh.test.en Data/translation_references/zh-en.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: zh->en test bleu: $zh2en"
