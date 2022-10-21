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
     IdentityFile ~/.ssh/${SSH_KEY_NAME}" > ${SSH_HOME}/config
# Copy public key, this is optional
scp -r ${USER_PATAS}@${PATAS}:${PATAS_SSH_HOME}/${SSH_KEY_NAME}.pub ${SSH_HOME}/
echo "Done."
echo

echo "Transfer data..."
# scp is 4x faster than rsync, so we use it for the first time.
if [ ! -d "~/Data" ]; then
     scp -r ${USER_PATAS}@${PATAS}:/projects/unmt/communication-translation/Data ~/
else
     rsync -azP ${USER_PATAS}@${PATAS}:/projects/unmt/communication-translation/Data ~/
fi
echo "Done."
echo

echo "Clone repository..."
if [ ! -d "~/communication-translation" ]; then
     git clone git@github.com:CLMBRs/communication-translation.git
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
# For some reason, install from bash script doesn't work
echo "Note: 'pip install -r requirements.txt' needs to be run separately"
echo "Note: to install torch, use 'pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html'"
echo
