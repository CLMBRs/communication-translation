#!/bin/sh

HYPOTHESIS=$1
REFERENCE=$2
TOKENIZER=$3

sacrebleu $REFERENCE -i $HYPOTHESIS -m bleu -w 4 -b --tokenize $TOKENIZER
