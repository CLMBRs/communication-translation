#!/bin/bash
source activate unmt
git rev-parse HEAD

export PYTHONPATH=".:${PYTHONPATH}"

MODEL_DIR=$1
TRANSLATION_OUTPUT_DIR=$MODEL_DIR/../translation_results

mkdir -p $TRANSLATION_OUTPUT_DIR

# Example default run: test_en-zh_pipeline.sh Output/en-zh_pipeline/best_bleu

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: Calculating en->zh test translation scores"
python -u BackTranslation/translate.py --config Configs/translate/test_en2zh_translate.yaml --model_path $MODEL_DIR --output_dir $TRANSLATION_OUTPUT_DIR
# Special note that Chinese tokenization used here (zh) is different from the default setting (13a)
en2zh="$(./Tools/bleu.sh ${TRANSLATION_OUTPUT_DIR}/zh-en.en.test.zh Data/translation_references/zh-en.zh.test zh)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: en->zh test bleu: $en2zh"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: Calculating zh->en test translation scores"
python -u BackTranslation/translate.py --config Configs/translate/test_zh2en_translate.yaml --model_path $MODEL_DIR --output_dir $TRANSLATION_OUTPUT_DIR
zh2en="$(./Tools/bleu.sh ${TRANSLATION_OUTPUT_DIR}/zh-en.zh.test.en Data/translation_references/zh-en.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: zh->en test bleu: $zh2en"
