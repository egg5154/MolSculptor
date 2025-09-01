export SCRIPT_DIR=$(dirname $(readlink -f $0))
export MOLEDIT_CKPT_ROOT=$SCRIPT_DIR/../checkpoints/training/diffusion_transformer
mkdir -p $MOLEDIT_CKPT_ROOT

## train dit test
export THIS_TASK_DIR=demo
python -u $SCRIPT_DIR/../main_train_dit.py \
    --coordinator_address $1:8837 --num_processes $2 --rank $3 \
    --num_total_steps 2000000 \
    --name_list_path $SCRIPT_DIR/../data/diffusion_transformer/name_list.pkl \
    --logger_path $MOLEDIT_CKPT_ROOT/$THIS_TASK_DIR/Logs_train.txt \
    --random_seed 8888 \
    --np_random_seed 8888 \
    --save_ckpt_path $MOLEDIT_CKPT_ROOT/$THIS_TASK_DIR \
    --callback_steps 500 \
    --save_steps 50000 \
    --device_batch_size 32 \
    --config_path $SCRIPT_DIR/../checkpoints/diffusion_transformer/config.pkl
exit