#!/bin/bash
source activate unmt
git rev-parse HEAD

export PYTHONPATH=".:${PYTHONPATH}"

MODEL_DIR=$1
TRANSLATION_OUTPUT_DIR=$MODEL_DIR/../translation_results
# mkdir -p $TRANSLATION_OUTPUT_DIR

# IndicNLP_dir=./indic_nlp_library
# IndicNLP_resources_dir=./indic_nlp_resources
# export PYTHONPATH=$PYTHONPATH:${IndicNLP_dir}
# export INDIC_RESOURCES_PATH=${IndicNLP_resources_dir}

reference_dir=Data/translation_references

# Example default run: test_en-zh_comet.sh Output/en-zh_pipeline/translation_results

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_comet: Calculating en->zh test translation scores"
en2zh="$(comet-score -s ${reference_dir}/zh-en.en.test -t ${TRANSLATION_OUTPUT_DIR}/zh-en.en.test.zh -r ${reference_dir}/zh-en.zh.test --quiet)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_comet: en->zh test COMET: $en2zh"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: Calculating zh->en test translation scores"
zh2en="$(comet-score -s ${reference_dir}/zh-en.zh.test -t ${TRANSLATION_OUTPUT_DIR}/zh-en.zh.test.en -r ${reference_dir}/zh-en.en.test --quiet)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-zh_pipeline: zh->en test COMET: $zh2en"
