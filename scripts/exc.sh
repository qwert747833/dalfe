#!/bin/bash
export PYTHONPATH=/home/sdb/zhouyou/TimeCMA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0

data_path="exchange_rate"
seq_len=96
batch_size=16
pred_len=96
learning_rate=2e-5
channel=16
e_layer=2
d_layer=1
dropout_n=0.1

log_path="./Results/${data_path}/"
mkdir -p $log_path
log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 21 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 500 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &

# pred_len = 192
pred_len=192
learning_rate=2e-5
batch_size=16
channel=16
e_layer=2
d_layer=1
dropout_n=0.1

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 21 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 500 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &

# pred_len 336
pred_len=336
learning_rate=2e-5
channel=16
batch_size=16
e_layer=2
d_layer=1
dropout_n=0.1

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 21 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 500 \
  --seed 2024 \
  --channel $channel \
  --head 8 \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &

# pred_len 720
pred_len=720
learning_rate=2e-5
channel=32
batch_size=16
e_layer=2
d_layer=1
dropout_n=0.1

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 21 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 500 \
  --seed 2024 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &