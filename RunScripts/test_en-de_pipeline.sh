#!/bin/bash
source activate unmt
git rev-parse HEAD

export PYTHONPATH=".:${PYTHONPATH}"

MODEL_DIR=$1
TRANSLATION_OUTPUT_DIR=$MODEL_DIR/../translation_results

mkdir -p $TRANSLATION_OUTPUT_DIR

# Example default run: test_en-de_pipeline.sh Output/en-de_pipeline/best_bleu

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: Calculating en->de test translation scores"
python -u BackTranslation/translate.py --config Configs/translate/test_en2de_translate.yaml --model_path $MODEL_DIR --output_dir $TRANSLATION_OUTPUT_DIR
en2de="$(./Tools/bleu.sh ${TRANSLATION_OUTPUT_DIR}/de-en.en.test.de Data/translation_references/de-en.de.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: en->de test bleu: $en2de"

echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: Calculating de->en test translation scores"
python -u BackTranslation/translate.py --config Configs/translate/test_de2en_translate.yaml --model_path $MODEL_DIR --output_dir $TRANSLATION_OUTPUT_DIR
de2en="$(./Tools/bleu.sh ${TRANSLATION_OUTPUT_DIR}/de-en.de.test.en Data/translation_references/de-en.en.test 13a)"
echo "$(date +'%Y-%m-%d %H:%M:%S') test_en-de_pipeline: de->en test bleu: $de2en"
