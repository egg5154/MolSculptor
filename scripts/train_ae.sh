#!/usr/bin/env bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
export MOLEDIT_CKPT_ROOT=$SCRIPT_DIR/../checkpoints/training/auto_encoder
mkdir -p $MOLEDIT_CKPT_ROOT

python -u $SCRIPT_DIR/../main_pretrain_ae.py \
    --coordinator_address $1:8837 --num_processes $2 --rank $3 \
    --num_total_epochs 10 \
    --name_list_path $SCRIPT_DIR/../data/auto_encoder/name_list.pkl \
    --logger_path $MOLEDIT_CKPT_ROOT/Logs_train.txt \
    --random_seed 8888 \
    --np_random_seed 8888 \
    --save_ckpt_path $MOLEDIT_CKPT_ROOT \
    --start_step 0 \
    --callback_steps 20 \
    --pre_load_steps 20 \
    --save_steps 5000 \
    --device_batch_size 1 \
    --config_path $SCRIPT_DIR/../checkpoints/auto_encoder/config.pkl
exit