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

# Example default run: test_en-si_pipeline.sh Output/en-si_pipeline/best_bleu

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-si_pipeline: Calculating en->si test translation scores"
# Tokenize both the reference(gold answer) and hypothesis(prediction)
# if [ ! -f ${reference_dir}/si-en.si.test.tok ]; then
#     echo "Tokenizing reference"
#     python ${IndicNLP_dir}/indicnlp/cli/cliparser.py tokenize ${reference_dir}/si-en.si.test ${reference_dir}/si-en.si.test.tok -l sin
# fi
# python ${IndicNLP_dir}/indicnlp/cli/cliparser.py tokenize ${TRANSLATION_OUTPUT_DIR}/si-en.en.test.si ${TRANSLATION_OUTPUT_DIR}/si-en.en.test.si.tok -l sin
# Calculate the bleu score (`none` = no tokenizer is used)
en2si="$(comet-score -s ${reference_dir}/si-en.en.test -t ${TRANSLATION_OUTPUT_DIR}/si-en.en.test.si -r ${reference_dir}/si-en.si.test)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-si_pipeline: en->si test COMET: $en2si"

# echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-si_pipeline: Calculating si->en test translation scores"
# si2en="$()"
# echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-si_pipeline: si->en test bleu: $si2en"
