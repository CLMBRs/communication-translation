#!/bin/bash
SSH_HOME=~/.ssh
USER_PATAS=zeyuliu2
PATAS=patas.ling.washington.edu
PATAS_SSH_HOME="/home2/${USER_PATAS}/.ssh"
SSH_KEY_NAME=leo


echo "Make .ssh directory"
cd 
mkdir -p ${SSH_HOME}
echo "Done."
echo


echo "Initialize ssh..."
scp -r ${USER_PATAS}@${PATAS}:${PATAS_SSH_HOME}/${SSH_KEY_NAME} ${SSH_HOME}/
# Intialize config
echo "Host *
     AddKeysToAgent yes
     XAuthLocation /opt/X11/bin/xauth
     IdentityFile ~/.ssh/${SSH_KEY_NAME}" >> ${SSH_HOME}/config
chmod 600 ${SSH_HOME}/config
# Copy public key, this is optional
scp -r ${USER_PATAS}@${PATAS}:${PATAS_SSH_HOME}/${SSH_KEY_NAME}.pub ${SSH_HOME}/
echo "Done."
echo

echo "Transfer data..."
# scp is 4x faster than rsync, so we use it for the first time.
if [ ! -d "~/Data" ]; then
     scp -r ${USER_PATAS}@${PATAS}:/projects/unmt/communication-translation-old/Data ~/
     scp -r ${USER_PATAS}@${PATAS}:/projects/unmt/communication-translation-old/Output/mbart_lm_lr6e-6 ~/communication-translation/Output/
else
     rsync -azP -e ssh ${USER_PATAS}@${PATAS}:/projects/unmt/communication-translation-old/Data ~/
     rsync -azP -e ssh ${USER_PATAS}@${PATAS}:/projects/unmt/communication-translation-old/Output/mbart_lm_lr6e-6 ~/communication-translation/Output/
fi
echo "Done."
echo

echo "Clone repositories..."
if [ ! -d "~/communication-translation" ]; then
     echo "Clone our repositories..."
     git clone git@github.com:CLMBRs/communication-translation.git
     cd communication-translation
     echo "Clone repositories for tokenizing si and ne ..."
     # clone tokenizer for si and ne
     git clone https://github.com/anoopkunchukuttan/indic_nlp_library.git
     git clone https://github.com/anoopkunchukuttan/indic_nlp_resources.git
     cd ../
fi
cd communication-translation/
if [ ! -d "./DataLink" ]; then
     ln -s ~/Data DataLink
fi
cd 
echo "Done."
echo

echo "Create conda environment..."
conda create --name unmt python=3.9
conda init
echo "Done."
echo 'export PYTHONPATH="~/communication-translation:$PYTHONPATH"' >> ~/.bashrc
# For some reason, install from bash script doesn't work
echo "You need to install our packages manually:"
echo "1. To install torch with CUDA 10.2, use 'pip install torch==1.12.1+cu102 --extra-index-url https://download.pytorch.org/whl/cu102'"
echo "2. pip install -r requirements.txt"
echo


echo "For si/ne tokenization, you need to install their packages manually:"
echo "1. Run 'cd indic_nlp_library'"
echo "2. 'pip install -r requirements.txt'"
echo