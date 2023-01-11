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

# Example default run: test_en-ne_comet.sh Output/en-ne_pipeline/best_bleu

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_comet: Calculating en->ne test translation scores"
# Tokenize both the reference(gold answer) and hypothesis(prediction)
# if [ ! -f ${reference_dir}/ne-en.ne.test.tok ]; then
#     echo "Tokenizing reference"
#     python ${IndicNLP_dir}/indicnlp/cli/cliparser.py tokenize ${reference_dir}/ne-en.ne.test ${reference_dir}/ne-en.ne.test.tok -l nep
# fi
# python ${IndicNLP_dir}/indicnlp/cli/cliparser.py tokenize ${TRANSLATION_OUTPUT_DIR}/ne-en.en.test.ne ${TRANSLATION_OUTPUT_DIR}/ne-en.en.test.ne.tok -l nep
# Calculate the bleu score (`none` = no tokenizer is used)
en2ne="$(comet-score -s ${reference_dir}/ne-en.en.test -t ${TRANSLATION_OUTPUT_DIR}/ne-en.en.test.ne -r ${reference_dir}/ne-en.ne.test)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_comet: en->ne test COMET: $en2ne"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_comet: Calculating ne->en test translation scores"
ne2en="$(comet-score -s ${reference_dir}/ne-en.ne.test -t ${TRANSLATION_OUTPUT_DIR}/ne-en.ne.test.en -r ${reference_dir}/ne-en.en.test)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_comet: ne->en test COMET: $ne2en"
