# Loading dataset used in Sec 5.1 in mBART

## General template

If you are running `backtranslate.py`, consider modify field: `val_dataset_script` and `lang_pair`

```python
from datasets import load_dataset
load_dataset(
            args.val_dataset_script, args.lang_pair, split="validation"
        )
```

## En-ne

```python
load_dataset("[path to]/flores/flores.py", "neen")
```

## En-si

```python
load_dataset("[path to]/flores/flores.py", "sien")
```

## En-zh

```python
load_dataset("[path to]/wmt19/wmt19.py", "zh-en")
```

## En-de

```python
load_dataset("[path to]/wmt19/wmt19.py", "de-en")
```

## En-ro

```python
load_dataset("[path to]/wmt19/wmt19.py", "ro-en")
```

# Measure Translation quality

# Evaluate results of Nepali/Sinhala with IndicNLP


First clone and follow set up the [IndicNLP repository](https://github.com/anoopkunchukuttan/indic_nlp_library) (See details in the repository README). Then, follow the instructions below:

nep = Nepali,  sin = Sinhala

```bash
# tokenize reference file (assume untokenized)
python <IndicNLP dir>/indicnlp/cli/cliparser.py tokenize <path to reference file>  <path to *tokenized* reference file> -l nep/sin

# tokenize output/hypothesis file (assume untokenized)
python <IndicNLP dir>/indicnlp/cli/cliparser.py tokenize <path to output file> <path to *tokenized* output file> -l nep/sin

# use `none` to denote input files are already tokenized
<communication-translation dir>/Tools/bleu.sh <path to *tokenized* output file> <path to *tokenized* reference file> none
```