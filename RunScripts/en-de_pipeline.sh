#!/bin/bash
source activate unmt

OUTPUT_DIR=$1
BT_INIT_CONFIG=$2
CAPTIONS_CONFIG=$3
EC_CONFIG=$4
BT_SECONDARY_CONFIG=$5

# Do initial (short) backtranslation
python -u BackTranslation/backtranslate.py --config Configs/${BT_INIT_CONFIG}.yml --backtranslated_dir ${OUTPUT_DIR}

# Do caption training
python -u -m EC_finetune --config Configs/${CAPTIONS_CONFIG}.yml --ec_dir ${OUTPUT_DIR}

# Do EC
python -u -m EC_finetune --config Configs/${EC_CONFIG}.yml --ec_dir ${OUTPUT_DIR}

# Get translation validation scores after EC training
cp ${OUTPUT_DIR}/bt_init/de-en.en.val ${OUTPUT_DIR}
cp ${OUTPUT_DIR}/bt_init/de-en.de.val ${OUTPUT_DIR}
python -u BackTranslation/translate.py --config Configs/en2de_translate.yml \
    --model_path ${OUTPUT_DIR}/ec \
    --output_dir ${OUTPUT_DIR}
python -u BackTranslation/translate.py --config Configs/de2en_translate.yml \
    --model_path ${OUTPUT_DIR}/ec \
    --output_dir ${OUTPUT_DIR}
en2de=$(./Tools/bleu.sh ${OUTPUT_DIR}/de-en.en.val.de ${OUTPUT_DIR}/de-en.de.val 13a)
de2en=$(./Tools/bleu.sh ${OUTPUT_DIR}/de-en.de.val.en ${OUTPUT_DIR}/de-en.en.val 13a)
echo "Post EC: en->de validation bleu: $en2de; de->en validation bleu: $de2en"

# Do rest of backtranslation
python -u BackTranslation/backtranslate.py --config Configs/${BT_SECONDARY_CONFIG}.yml --seed_override 2  --backtranslated_dir ${OUTPUT_DIR}
