#!/bin/bash

pip install sklearn
export  DATA_DIR=/home/gad/data/raw/
mkdir -p $DATA_DIR
gsutil -m cp gs://draw-me-a-sheep/images/* ./data/raw/
git clone  -b tpu https://github.com/doitintl/ML-on-Cloud-Examples.git
cd ML-on-Cloud-Examples/tpu/classifier/
python train_cnn.py \
    --tpu=$TPU_NAME \
    --use_tpu=False \
    --iterations=500 \
    --train_steps=20000\
    --data_dir=$DATA_DIR \
     --model_dir=gs://tpu-demo-xsb62/output/



