# UNMT_wEye: Unsupervised Neural Machine Translation with image-selection as finetuning signal
This repository contains the code for finetuning a pretrained multilingual model
with an emergent-communication communication task grounded in image recognition.

The code is loosely based on the work from the following paper

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

## Acknowledgements
Part of the code is based on 
[Translagent](https://github.com/facebookresearch/translagent). 

The datasets for our experiments include [MS COCO](http://cocodataset.org/#home)
for Emergent Communication pretraining, 
