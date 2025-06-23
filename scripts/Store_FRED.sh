export PYTHONPATH=/home/sdb/zhouyou/TimeCMA:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=1

num_nodes=7
input_len=36
output_len=24
# 
divides=("train" "test" "val")
datasets=("FRED")
for dataset in "${datasets[@]}"; do
    for divide in "${divides[@]}"; do
        log_file="./Results/emb_logs/${dataset}/_${divide}.log"
        python storage/store_emb.py --data_path "$dataset" --divide $divide --num_nodes $num_nodes --input_len $input_len --output_len $output_len 
    done
done