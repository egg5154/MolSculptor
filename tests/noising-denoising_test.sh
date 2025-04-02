#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
python -u $SCRIPT_DIR/noising-denoising_test.py \
    --config_path $SCRIPT_DIR/../checkpoints/diffusion-transformer/config.pkl \
    --params_path $SCRIPT_DIR/../checkpoints/diffusion-transformer/dit_params_step1000000.pkl \
    --vae_config_path $SCRIPT_DIR/../checkpoints/auto-encoder/config.pkl \
    --vae_params_path $SCRIPT_DIR/../checkpoints/auto-encoder/ae_params_step265000.pkl \
    --alphabet_path $SCRIPT_DIR/../train/smiles_alphabet.pkl \
    --random_seed 8888 \
    --sampling_method beam \
    --beam_size 4 \
    --init_molecule_path $SCRIPT_DIR/init-molecule/init_search_molecule.pkl \
    --save_path $SCRIPT_DIR/init-molecule/noising-denoising_test.pkl
exit