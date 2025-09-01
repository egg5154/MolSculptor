# #!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
python -u $SCRIPT_DIR/noising-denoising_test.py \
    --params_path $SCRIPT_DIR/../checkpoints/diffusion_transformer/params.pkl \
    --config_path $SCRIPT_DIR/../checkpoints/diffusion_transformer/config.pkl \
    --vae_config_path $SCRIPT_DIR/../checkpoints/auto_encoder/config.pkl \
    --vae_params_path $SCRIPT_DIR/../checkpoints/auto_encoder/params.pkl \
    --alphabet_path $SCRIPT_DIR/../train/smiles_alphabet.pkl \
    --random_seed 8888 \
    --sampling_method beam \
    --beam_size 4 \
    --init_molecule_path $1/init_search_molecule.pkl \
    --save_path $1/noising-denoising_test.pkl
exit