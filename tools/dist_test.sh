CONFIG=$1
CHECKPOINT=$2
GPU=$3
BATCH_SIZE=${BATCH_SIZE:-32}

python -m torch.distributed.launch \
    --nproc_per_node=$GPU \
    --master_port=$((RANDOM + 10000)) \
    tools/test_net.py \
    --config-file $CONFIG \
    MODEL.WEIGHT $CHECKPOINT \
    TEST.IMS_PER_BATCH $BATCH_SIZE