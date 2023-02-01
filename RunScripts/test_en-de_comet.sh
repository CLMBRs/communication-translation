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

# Example default run: test_en-de_comet.sh Output/en-de_pipeline/translation_results

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_comet: Calculating en->de test translation scores"
en2de="$(comet-score -s ${reference_dir}/de-en.en.test -t ${TRANSLATION_OUTPUT_DIR}/de-en.en.test.de -r ${reference_dir}/de-en.de.test --quiet)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_comet: en->de test COMET: $en2de"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: Calculating de->en test translation scores"
de2en="$(comet-score -s ${reference_dir}/de-en.de.test -t ${TRANSLATION_OUTPUT_DIR}/de-en.de.test.en -r ${reference_dir}/de-en.en.test --quiet)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: de->en test COMET: $de2en"
