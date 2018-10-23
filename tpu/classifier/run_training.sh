#!/bin/bash

pip install sklearn
mkdir -p ./data/raw/
gsutil -m cp gs://draw-me-a-sheep/images/* ./data/raw/
gsutil cp gs://tpu-demo-xsb62/code/* ./
python train_cnn.py \
    --tpu=$TPU_NAME \
    --use_tpu=True \
    --iterations=500 \
    --train_steps=20000\
    --data_dir=./data/raw/ \
     --model_dir=gs://tpu-demo-xsb62/output/



