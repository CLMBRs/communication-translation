list_of_dir="""
Output/en-zh/t2i/resnet+transformer/seed2/bt_sec_from-pretrained/translation_results
"""
for i in $list_of_dir 
do 
    echo $i 
    bash retrieve.sh ${i}
    echo
done


# bash retrieve.sh Output/en-si/i2i/resnet+transformer/seed1/bt_sec_from-last/translation_results/
# bash retrieve.sh Output/en-si/i2i/resnet+transformer/seed2/bt_sec_from-last/translation_results/
# bash retrieve.sh Output/en-si/i2i/resnet+transformer/seed3/bt_sec_from-last/translation_results/

# bash retrieve.sh Output/en-si/i2i/resnet+transformer/seed1/bt_sec_from-last/translation_results/
# bash retrieve.sh Output/en-si/i2i/resnet+transformer/seed2/bt_sec_from-last/translation_results/
# bash retrieve.sh Output/en-si/i2i/resnet+transformer/seed3/bt_sec_from-last/translation_results/

# bash retrieve.sh Output/en-si/i2i/resnet+transformer/seed1/bt_sec_from-last/translation_results/
# bash retrieve.sh Output/en-si/i2i/resnet+transformer/seed2/bt_sec_from-last/translation_results/
# bash retrieve.sh Output/en-si/i2i/resnet+transformer/seed3/bt_sec_from-last/translation_results/
