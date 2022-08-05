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

## Dependencies
The source code is built aroud PyTorch, and has the following main dependencies:

- Python 3.9
- PyTorch >=1.7.0
- transformers 4.0.1

For more extensive dependencies, see `requirements.txt`.

    pip install -r requirements.txt

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

## Data
COCO image features are obtained from [Translagent](https://github.com/facebookresearch/translagent).

## Pipeline
To run code in this package, you must first do the following steps:
1. Create a Python 3.9 Conda virtual environment: `conda create -n unmt python=3.9`
1. Start the new environment: `conda activate unmt`
1. Install this package using developer mode from the top level directory (`communication_translation`): `pip install -e .`

To obtain the latest results of this project, go to the `communication-translation` folder and
run the relevant script from [RunScripts](/RunScripts).

## Acknowledgements
Part of the code is based on 
[Translagent](https://github.com/facebookresearch/translagent). 

The datasets for our experiments include [MS COCO](http://cocodataset.org/#home)
for Emergent Communication pretraining, 
