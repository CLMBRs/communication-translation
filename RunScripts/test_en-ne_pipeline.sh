#!/bin/bash
source activate unmt
git rev-parse HEAD

export PYTHONPATH=".:${PYTHONPATH}"

MODEL_DIR=$1
TRANSLATION_OUTPUT_DIR=$MODEL_DIR/../translation_results
mkdir -p $TRANSLATION_OUTPUT_DIR

IndicNLP_dir=./indic_nlp_library
IndicNLP_resources_dir=./indic_nlp_resources
export PYTHONPATH=$PYTHONPATH:${IndicNLP_dir}
export INDIC_RESOURCES_PATH=${IndicNLP_resources_dir}

reference_dir=Data/translation_references

# Example default run: test_en-ne_pipeline.sh Output/en-ne_pipeline/best_bleu

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_pipeline: Calculating en->ne test translation scores"
python -u BackTranslation/translate.py --config Configs/translate/test_en2ne_translate.yaml --model_path $MODEL_DIR --output_dir $TRANSLATION_OUTPUT_DIR
# Tokenize both the reference(gold answer) and hypothesis(prediction)
if [ ! -f ${reference_dir}/ne-en.ne.test.tok ]; then
    echo "Tokenizing reference"
    python ${IndicNLP_dir}/indicnlp/cli/cliparser.py tokenize ${reference_dir}/ne-en.ne.test ${reference_dir}/ne-en.ne.test.tok -l nep
fi
python ${IndicNLP_dir}/indicnlp/cli/cliparser.py tokenize ${TRANSLATION_OUTPUT_DIR}/ne-en.en.test.ne ${TRANSLATION_OUTPUT_DIR}/ne-en.en.test.ne.tok -l nep
# Calculate the bleu score (`none` = no tokenizer is used)
en2ne="$(./Tools/bleu.sh ${TRANSLATION_OUTPUT_DIR}/ne-en.en.test.ne.tok ${reference_dir}/ne-en.ne.test.tok none)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_pipeline: en->ne test bleu: $en2ne"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_pipeline: Calculating ne->en test translation scores"
python -u BackTranslation/translate.py --config Configs/translate/test_ne2en_translate.yaml --model_path $MODEL_DIR --output_dir $TRANSLATION_OUTPUT_DIR
ne2en="$(./Tools/bleu.sh ${TRANSLATION_OUTPUT_DIR}/ne-en.ne.test.en ${reference_dir}/ne-en.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-ne_pipeline: ne->en test bleu: $ne2en"
