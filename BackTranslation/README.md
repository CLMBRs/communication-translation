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
