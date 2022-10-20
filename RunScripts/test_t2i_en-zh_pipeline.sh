#!/bin/bash
source activate unmt

OUTPUT_DIR=$1

# Example default run: test_t2i_en-zh_pipeline.sh Output/en-zh_tec

echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-zh_pipeline: Calculating en->zh test translation scores"
python -u BackTranslation/translate.py --config Configs/t2i/test_en2zh_translate.yml
en2zh="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/zh-en.en.test.zh Data/translation_references/zh-en.zh.test zh)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-zh_pipeline: en->zh test bleu: $en2zh"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-zh_pipeline: Calculating zh->en test translation scores"
python -u BackTranslation/translate.py --config Configs/t2i/test_zh2en_translate.yml
zh2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/zh-en.zh.test.en Data/translation_references/zh-en.en.test zh)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_t2i_en-zh_pipeline: zh->en test bleu: $zh2en"
