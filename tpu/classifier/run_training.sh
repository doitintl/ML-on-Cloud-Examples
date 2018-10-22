#!/bin/bash

pip install sklearn
mkdir -p ./data/raw/
gsutil -m cp gs://draw-me-a-sheep/images/* ./data/raw/
gsutil cp gs://tpu-demo-xsb62/* ./
python train_cnn.py \
    --tpu=$TPU_NAME \
    --use_tpu=True \
    --iterations=500 \
    --train_steps=2000



