#!/bin/bash

pip install sklearn
mkdir -p ./data/raw/
gsutil -m cp gs://draw-me-a-sheep/images/* ./data/raw/
gid clone https://github.com/doitintl/ML-on-Cloud-Examples.git
cd tpu/classifier/
python train_cnn.py \
    --tpu=$TPU_NAME \
    --use_tpu=False \
    --iterations=500 \
    --train_steps=20000\
    --data_dir=./data/raw/ \
     --model_dir=gs://tpu-demo-xsb62/output/



