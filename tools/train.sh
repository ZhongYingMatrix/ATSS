CONFIG=$1
GPU=$2

python -m torch.distributed.launch \
    --nproc_per_node=$GPU \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net.py \
    --config-file $CONFIG