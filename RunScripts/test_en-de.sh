cp /projects/unmt/communication-translation/Data/translation_references/de-en.* Output/en-de_pipeline
python -u BackTranslation/translate.py --config Configs/en2de_translate.yml --output_dir Output/en-de_pipeline --model_path <PATH_TO_MODEL>
python -u BackTranslation/translate.py --config Configs/de2en_translate.yml --output_dir Output/en-de_pipeline --model_path <PATH_TO_MODEL>
./Tools/bleu.sh Output/en-de_pipeline/de-en.en.test.de Output/en-de_pipeline/de-en.de.test 13a
./Tools/bleu.sh Output/en-de_pipeline/de-en.de.test.en Output/en-de_pipeline/de-en.en.test 13a