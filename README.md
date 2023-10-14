# Emergent Communication Fine-tuning for Pre-trained Language Models

This repository contains the code for finetuning a pretrained multilingual model
with an image-grounded Emergent Communication task. This is the official
repository for the ICLR EmeComm Workshop Paper
[Emergent Communication Fine-tuning (EC-FT) for Pretrained Language Models](https://openreview.net/forum?id=SUqrM7WR7W5)
and the July 2022 pre-print "Learning to Translate by Learning to Communicate".

The image-to-image results (i2i-ec) from the July 2022 preprint should be
replicable on the tag [preprint_jul22_i2i-ec](https://github.com/CLMBRs/communication-translation/releases/tag/preprint_jul22_i2i-ec).
The text-to-image results (t2i-ec) should be replicable on
[preprint_jul22_t2i-ec](https://github.com/CLMBRs/communication-translation/releases/tag/preprint_jul22_t2i-ec).

## Code acknowledgement

The code was loosely based on the work of the following paper

Yaoyiran Li, Edoardo Maria Ponti, Ivan Vulić, and Anna Korhonen. 2020.
*Emergent Communication Pretraining for Few-Shot Machine Translation*. In
Proceedings of the 28th International Conference on Computational Linguistics
(COLING 2020). [LINK](https://www.aclweb.org/anthology/2020.coling-main.416.pdf)

and is under development by the **University of Washington CLMBR Lab**, under
Shane Steinert-Threlkeld.

## Installation

The source code is built aroud PyTorch, and has the following main dependencies:

- Python 3.9
- transformers 4.0.1


For more extensive dependencies, see `requirements.txt`.
```
    conda create -n unmt python=3.9
    pip install -r requirements.txt
    pip install torch==1.12.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
```


**Important Note:** We develop this project with `torch==1.12.1+cu102`, please make sure this is package is used for reproducibility.

## Config / Experiments

To obtain the latest results of this project, go to the `communication-translation` folder and
run the relevant script from [RunScripts](/RunScripts).


This project uses structured configs implemented by [`hydra`](https://hydra.cc/docs/intro/), located in `Configs` directory. We will briefly explain how configs are organized in this project and we refer user to original `hydra` documentation for understanding what "structure configs" means.

Our pipeline mainly consists of two parts: backtranslation(`Configs/backtranslate`, "BT") and emergent communication(`Configs/ec`, "EC"). Backtranslation follows the iterative backtranslation process in [mBART paper](https://arxiv.org/abs/2001.08210) but applied to different language pairs. 

Backtranslation(BT) is the main bulk part of the experiments and BT in all experiments run for the same number of steps. **One could view EC training as a super light-weight training (about 30min of EC and 12hr of BT) inserted into the backtranslation process.** Given a language pair and image embedding source, different experiments mainly vary across two dimensions: **1.** Where EC is inserted in the process of BT **2.** Which type of EC is inserted (T2I or I2I).

```
|===== BT =====||===== Optional: EC ======||=========== BT ============|
```

With that in mind, we organize configs as follows. We will use BT as an example:

```
Configs/backtranslate/
├── bt_baseline.yaml # baseline, that only do BT
├── data  # pointing to different language data files
│   ├── en-de.yaml
│   ├── en-ne.yaml
│   ├── en-si.yaml
│   └── en-zh.yaml
└── train_eval # different training configs for backtranslation 
│   ├── baseline.yaml
│   ├── initial.yaml
│   └── secondary.yaml
│   # different configs for different experiments, 
│   # each configs essentially combine sub-configs in `data` and `train_eval`
├── i2i_bt_initial.yaml  
├── i2i_bt_no_initBT.yaml
├── i2i_bt_secondary.yaml
├── t2i_bt_initial.yaml
├── t2i_bt_secondary.yaml
└── t2i_bt.yaml
```
Since we ran our experiments on server managed by [condor](https://courses.washington.edu/ling571/ling571_WIN2017/orientation.pdf), we included many `*.cmd` files at the repository root.

## Style Guide

Source code can be largely automatically formatted using yapf. Make sure you
have yapf installed (it is included in requirements.txt).

    pip install yapf

The repository style can be changed as we need, but for now the configuration
can be found in `setup.cfg`. To automatically format source code in place, use
the following command:

    yapf -ir src_file_or_directory

We recommend running this command after you add any code, and *before you
commit*.

Please also follow other style best practices that yapf does not enforce:

- Always break up lines over 80 characters (*in most text editors you can
display a ruler to check*)
- Name variables with **full, descriptive words**, space permitting
- Include one blank line at the end of every file
- Organize imports into the following three groups, alphabetizing within each
group (and within group, put Python library imports before external package
ones)
  - `import a`
  - `import a as b`
  - `from a import b`
- **Comment any code you add**
  - "Imperative" style is preferred, e.g.\
    `# save variable to cache`

## Data / Model Release

COCO image features are obtained from [Translagent](https://github.com/facebookresearch/translagent).

We publicize our data and model at [huggingface hub](https://huggingface.co/CLMBR/ec-unmt). 
The data is under `Data/`; we additionally use a language model to regularize training, the used model is essentially a finetuned mBART decoder (under `Output/mbart_lm_lr6e-6`).

## Acknowledgements

Part of the code is based on
[Translagent](https://github.com/facebookresearch/translagent).

The datasets for our experiments include [MS COCO](http://cocodataset.org/#home)
for Emergent Communication pretraining,
