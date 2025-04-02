#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
python -u $SCRIPT_DIR/../../diff_evo_denovo_pi3k.py \
    --params_path $SCRIPT_DIR/../../checkpoints/diffusion-transformer/dit_params_denovo.pkl \
    --config_path $SCRIPT_DIR/../../checkpoints/diffusion-transformer/config_denovo.pkl \
    --logger_path $SCRIPT_DIR/test/Logs.txt \
    --save_path $SCRIPT_DIR/test \
    --random_seed 6666 \
    --np_random_seed 6666 \
    --total_step 30 \
    --device_batch_size 128 \
    --n_replicate 8 \
    --num_latent_tokens 16 \
    --dim_latent 32 \
    --eq_steps 10 \
    --beam_size 4 \
    --sampling_method beam \
    --vae_config_path $SCRIPT_DIR/../../checkpoints/auto-encoder/config.pkl \
    --vae_params_path $SCRIPT_DIR/../../checkpoints/auto-encoder/ae_params_denovo.pkl \
    --alphabet_path $SCRIPT_DIR/../../train/smiles_alphabet.pkl \
    --init_molecule_path $SCRIPT_DIR/padding_molecule_propane.pkl
exit