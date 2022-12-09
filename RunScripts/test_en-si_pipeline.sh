#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
IndicNLP_dir=../indic_nlp_library
IndicNLP_resources_dir=../indic_nlp_resources
export PYTHONPATH=$PYTHONPATH:${IndicNLP_dir}
export INDIC_RESOURCES_PATH=${IndicNLP_resources_dir}

reference_dir=Data/translation_references

# Example default run: test_en-si_pipeline.sh Output/en-si_pipeline Output/en-si_pipeline/best_bleu

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-si_pipeline: Calculating en->si test translation scores"
python -u BackTranslation/translate.py --config Configs/test_en2si_translate.yml
# Tokenize both the reference(gold answer) and hypothesis(prediction)
if [ ! -f ${reference_dir}/si-en.si.test.tok ]; then
    echo "Tokenizing reference"
    python ${IndicNLP_dir}/indicnlp/cli/cliparser.py tokenize ${reference_dir}/si-en.si.test ${reference_dir}/si-en.si.test.tok -l sin
fi
python ${IndicNLP_dir}/indicnlp/cli/cliparser.py tokenize ${OUTPUT_DIR}/translation_results/si-en.en.test.si ${OUTPUT_DIR}/translation_results/si-en.en.test.si.tok -l sin
# Calculate the bleu score (`none` = no tokenizer is used)
en2si="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/si-en.en.test.si ${reference_dir}/si-en.si.test none)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-si_pipeline: en->si test bleu: $en2si"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-si_pipeline: Calculating si->en test translation scores"
python -u BackTranslation/translate.py --config Configs/test_si2en_translate.yml
si2en="$(./Tools/bleu.sh ${OUTPUT_DIR}/translation_results/si-en.si.test.en ${reference_dir}/si-en.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-si_pipeline: si->en test bleu: $si2en"
