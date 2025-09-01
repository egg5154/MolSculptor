#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
export MOLEDIT_CKPT_ROOT=$SCRIPT_DIR/../checkpoints/training/affinity_predictor
mkdir -p $MOLEDIT_CKPT_ROOT

## train dit test
export THIS_TASK_DIR=demo
mkdir -p $MOLEDIT_CKPT_ROOT/$THIS_TASK_DIR
python -u $SCRIPT_DIR/../main_train_affinity_predictor.py \
    --save_ckpt_path $MOLEDIT_CKPT_ROOT/$THIS_TASK_DIR \
    --save_logger_path $MOLEDIT_CKPT_ROOT/$THIS_TASK_DIR/Logs.txt \
    --n_total_steps 100000 \
    --batch_size 256 \
    --callback_steps 20 \
    --save_steps 1000 \
    --random_seed 8888 \
    --name_list_path $SCRIPT_DIR/../data/affinity_predictor/name_list.pkl \
    --config_path $SCRIPT_DIR/../checkpoints/affinity_predictor/config.pkl