#!/bin/bash
set -x

#conda create --name grocery_seg python=3.12
#conda activate grocery_seg

python -m venv .
source ./bin/activate
python -m ensurepip --upgrade

pip install --upgrade pip setuptools wheel

pip install opencv-python opencv-contrib-python
pip install 'numpy<2.0'
pip install 'ultralytics<8.3.40' #no mining allowed!
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install fastapi
read -p "Press any key to continue"

cd app
python main.py