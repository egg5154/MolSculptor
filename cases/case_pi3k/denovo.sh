#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
export CUDA_VISIBLE_DEVICES=$2
export NPOP=4
export NREP=2
export NINIT_STEP=1
export NOPT_STEP=2
export SAVE_DIR=${SCRIPT_DIR}/results
mkdir -p $SAVE_DIR
python -u $SCRIPT_DIR/../../diff_denovo_selective.py \
    --params_path $SCRIPT_DIR/../../checkpoints/diffusion_transformer/params.pkl \
    --config_path $SCRIPT_DIR/../../checkpoints/diffusion_transformer/config.pkl \
    --logger_path $SAVE_DIR/denovo_pi3k_$(date "+%Y-%m-%d")_seed${1}_npop${NPOP}_nrep${NREP}_step${NSTEP}/Logs.txt \
    --save_path $SAVE_DIR/denovo_pi3k_$(date "+%Y-%m-%d")_seed${1}_npop${NPOP}_nrep${NREP}_step${NSTEP} \
    --random_seed $1 \
    --np_random_seed $1 \
    --init_step $NINIT_STEP \
    --opt_step $NOPT_STEP \
    --device_batch_size $NPOP \
    --n_replicate $NREP \
    --opt_t_min 125 \
    --opt_t_max 175 \
    --on_target_script $SCRIPT_DIR/dsdp_pi3k_alpha.sh \
    --off_target_scripts $SCRIPT_DIR/dsdp_pi3k_beta.sh $SCRIPT_DIR/dsdp_pi3k_delta.sh \
    --vae_config_path $SCRIPT_DIR/../../checkpoints/auto_encoder/config.pkl \
    --vae_params_path $SCRIPT_DIR/../../checkpoints/auto_encoder/params.pkl \
    --alphabet_path $SCRIPT_DIR/../../train/smiles_alphabet.pkl \
    --pocket_features_path $SCRIPT_DIR/pocket_features/pocket_features.pkl \
    --affinity_predictor_params_path $SCRIPT_DIR/../../checkpoints/affinity_predictor/params.pkl \
    --affinity_predictor_config_path $SCRIPT_DIR/../../checkpoints/affinity_predictor/config.pkl
exit