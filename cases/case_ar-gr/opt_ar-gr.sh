#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
python -u $SCRIPT_DIR/../../diff_evo_opt_dual.py \
    --params_path $SCRIPT_DIR/../../checkpoints/diffusion-transformer/dit_params_step265000.pkl \
    --config_path $SCRIPT_DIR/../../checkpoints/diffusion-transformer/config.pkl \
    --logger_path $SCRIPT_DIR/test/Logs.txt \
    --save_path $SCRIPT_DIR/test \
    --dsdp_script_path_1 $SCRIPT_DIR/dsdp_ar.sh \
    --dsdp_script_path_2 $SCRIPT_DIR/dsdp_gr.sh \
    --random_seed 8888 \
    --np_random_seed 8888 \
    --total_step 30 \
    --device_batch_size 4 \
    --t_min 75 \
    --t_max 125 \
    --n_replicate 1 \
    --num_latent_tokens 16 \
    --dim_latent 32 \
    --eq_steps 10 \
    --vae_config_path $SCRIPT_DIR/../../checkpoints/auto-encoder/config.pkl \
    --vae_params_path $SCRIPT_DIR/../../checkpoints/auto-encoder/ae_params_step265000.pkl \
    --alphabet_path $SCRIPT_DIR/../../train/smiles_alphabet.pkl \
    --init_molecule_path $SCRIPT_DIR/init_search_molecule.pkl \
    --sub_smiles 'NC(=O)c1cccc(S(=O)(=O)N2CCCc3ccccc32)c1'
exit