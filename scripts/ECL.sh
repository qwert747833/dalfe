#!/bin/bash
export PYTHONPATH=/home/sdb/zhouyou/TimeCMA:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1

data_path="ECL"
seq_len=96
batch_size=8
log_path="./Results/${data_path}/"
mkdir -p $log_path

# pred_len 96
pred_len=96
learning_rate=1e-3
channel=128
e_layer=3
d_layer=6
dropout_n=0.3

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 50 \
  --seed 8888 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &

# pred_len = 192
pred_len=192
learning_rate=1e-3
channel=128
e_layer=3
d_layer=6
dropout_n=0.3

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 50 \
  --seed 8888 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &

# pred_len 336
pred_len=336
learning_rate=1e-3
channel=128
e_layer=3
d_layer=6
dropout_n=0.1

log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
nohup python train.py \
  --data_path $data_path \
  --batch_size $batch_size \
  --num_nodes 7 \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --epochs 50 \
  --seed 8888 \
  --channel $channel \
  --learning_rate $learning_rate \
  --dropout_n $dropout_n \
  --e_layer $e_layer \
  --d_layer $d_layer > $log_file &

# pred_len 720
pred_len=720
learning_rate=1e-4
channel=128
e_layer=3
d_layer=6
dropout_n=0.1

# log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
# nohup python train.py \
#   --data_path $data_path \
#   --batch_size $batch_size \
#   --num_nodes 7 \
#   --seq_len $seq_len \
#   --pred_len $pred_len \
#   --epochs 50 \
#   --seed 8888 \
#   --channel $channel \
#   --learning_rate $learning_rate \
#   --dropout_n $dropout_n \
#   --e_layer $e_layer \
#   --d_layer $d_layer > $log_file &
log_file="${log_path}i${seq_len}_o${pred_len}_lr${learning_rate}_c${channel}_el${e_layer}_dl${d_layer}_dn${dropout_n}_bs${batch_size}.log"
torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 \
    --master_addr=127.0.0.1 --master_port=29500 \
    train.py \
    --data_path "$data_path" \
    --batch_size "$batch_size" \
    --num_nodes 7 \
    --seq_len "$seq_len" \
    --pred_len "$pred_len" \
    --epochs 50 \
    --seed 8888 \
    --channel "$channel" \
    --learning_rate "$learning_rate" \
    --dropout_n "$dropout_n" \
    --e_layer "$e_layer" \
    --d_layer "$d_layer" > "$log_file" 2>&1 &  # 将日志重定向到文件