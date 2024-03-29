#!/bin/bash
# update the repository
echo 'updating repository'
cd ~/IOFtrain
git reset --hard HEAD
git pull
# install dependencies
conda create -n IOFtrain python=3.9
conda activate IOFtrain
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda install -c conda-forge wandb
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
#conda install -c conda-forge imgaug
#conda install -c conda-forge albumentations
#conda install -c anaconda xmltodict

pip install --upgrade pip
pip install tensorflow
pip install tflite-model-maker
pip install pycocotools
pip install opencv-python
#pip install pascal_voc_writer
#pip install tqdm



curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler





