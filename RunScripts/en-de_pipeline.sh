#!/bin/bash
source activate unmt

EX_ABBR=clipL_slFz_transf
OUTPUT_DIR=Output/en-de_pipeline/bt_sec_${EX_ABBR}
BT_INIT_CONFIG=$2
CAPTIONS_CONFIG=en-de_captions
EC_CONFIG=en-de_ec
BT_SECONDARY_CONFIG=$5

# # Do initial (short) backtranslation
# python -u BackTranslation/backtranslate.py --config Configs/${BT_INIT_CONFIG}.yml

# Do caption training
python -u -m EC_finetune --config Configs/${CAPTIONS_CONFIG}.yml \
    --sender_freeze_override \
    --receiver_freeze_override \
    --output_dir_override Output/en-de_pipeline/captions_${EX_ABBR}

# Do EC
python -u -m EC_finetune --config Configs/${EC_CONFIG}.yml \
    --model_dir_override Output/en-de_pipeline/captions_${EX_ABBR} \
    --output_dir_override Output/en-de_pipeline/ec_${EX_ABBR}

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

# Do rest of backtranslation
python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml \
    --seed_override 2 \
    --model_dir_override Output/en-de_pipeline/ec_${EX_ABBR} \
    --output_dir_override ${OUTPUT_DIR}

# Do test
cp /projects/unmt/communication-translation/Data/translation_references/de-en.* ${OUTPUT_DIR}
python -u BackTranslation/translate.py --config Configs/en2de_translate.yml \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${OUTPUT_DIR}/best_bleu
python -u BackTranslation/translate.py --config Configs/de2en_translate.yml \
    --output_dir ${OUTPUT_DIR} \
    --model_path ${OUTPUT_DIR}/best_bleu
./Tools/bleu.sh ${OUTPUT_DIR}/de-en.en.test.de ${OUTPUT_DIR}/de-en.de.test 13a
./Tools/bleu.sh ${OUTPUT_DIR}/de-en.de.test.en ${OUTPUT_DIR}/de-en.en.test 13a
