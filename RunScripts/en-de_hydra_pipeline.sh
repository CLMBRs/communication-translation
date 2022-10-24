#!/bin/bash
EX_ABBR=resnet_recheck
OUTPUT_DIR=Output/en-de_pipeline/bt_sec_${EX_ABBR}
BT_INIT_CONFIG=bt_initial
CAPTIONS_CONFIG=captions
EC_CONFIG=ec
LANG=en-de
BT_SECONDARY_CONFIG=bt_secondary

# Do initial (short) backtranslation
python -u BackTranslation/backtranslate.py \
    +backtranslate=${BT_INIT_CONFIG} \
    backtranslate/data=${LANG}

# # Do caption training
python -u -m EC_finetune +ec=${CAPTIONS_CONFIG} \
    ec/language=${LANG} \
    ec.output_dir=Output/en-de_pipeline/captions_${EX_ABBR} \
    ec.model.model_name=Output/en-de_pipeline/bt_init/last \
    ec.model.freeze_sender=True \

# Do EC
python -u -m EC_finetune  +ec=${EC_CONFIG} \
    ec/language=${LANG} \
    ec.output_dir=Output/en-de_pipeline/ec_${EX_ABBR} \
    ec.model.model_name=Output/en-de_pipeline/captions_${EX_ABBR} \

# cp ${OUTPUT_DIR}/bt_init/de-en.en.val ${OUTPUT_DIR}
# cp ${OUTPUT_DIR}/bt_init/de-en.de.val ${OUTPUT_DIR}

# # Get translation validation scores after EC training
# python -u BackTranslation/translate.py --config Configs/en2de_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# python -u BackTranslation/translate.py --config Configs/de2en_translate.yml \
#     --model_path ${OUTPUT_DIR}/ec \
#     --output_dir ${OUTPUT_DIR}
# en2de=$(./Tools/bleu.sh ${OUTPUT_DIR}/de-en.en.val.de ${OUTPUT_DIR}/de-en.de.val 13a)
# de2en=$(./Tools/bleu.sh ${OUTPUT_DIR}/de-en.de.val.en ${OUTPUT_DIR}/de-en.en.val 13a)
# echo 'en to de score: '"$en2de"'; de to en score: '"$de2en"

# # Do rest of backtranslation
python -u BackTranslation/backtranslate.py \
    +backtranslate=${BT_SECONDARY_CONFIG} \
    backtranslate/data=${LANG} \
    backtranslate.train_eval.seed=2 \
    backtranslate.model_path=Output/en-de_pipeline/ec_${EX_ABBR} \
    backtranslate.output_dir=${OUTPUT_DIR}

# python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml \
#     --seed_override 2 \
#     --model_dir_override Output/en-de_pipeline/ec_${EX_ABBR} \
#     --output_dir_override ${OUTPUT_DIR}

# Do test
# cp /projects/unmt/communication-translation/Data/translation_references/de-en.* ${OUTPUT_DIR}
# python -u BackTranslation/translate.py --config Configs/test_en2de_translate.yml \
#     --output_dir ${OUTPUT_DIR} \
#     --model_path ${OUTPUT_DIR}/best_bleu
# python -u BackTranslation/translate.py --config Configs/test_de2en_translate.yml \
#     --output_dir ${OUTPUT_DIR} \
#     --model_path ${OUTPUT_DIR}/best_bleu

# cp Output/en-de_pipeline/translation_results/* ${OUTPUT_DIR}
# ./Tools/bleu.sh ${OUTPUT_DIR}/de-en.en.test.de ${OUTPUT_DIR}/de-en.de.test 13a
# ./Tools/bleu.sh ${OUTPUT_DIR}/de-en.de.test.en ${OUTPUT_DIR}/de-en.en.test 13a