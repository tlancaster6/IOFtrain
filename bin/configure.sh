#!/bin/bash
# update the repository
echo 'updating repository'
cd ~/IOFtrain
git reset --hard HEAD
git pull
# install dependencies
pip3 install -q tflite-support
pip3 install -q --use-deprecated=legacy-resolver tflite-model-maker
sudo apt-get install libportaudio2
pip3 install sounddevice
pip3 install numpy
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
sudo apt-get update
sudo apt-get install edgetpu-compiler
