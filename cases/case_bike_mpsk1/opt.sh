#!/bin/bash

export SCRIPT_DIR=$(dirname $(readlink -f $0))
export CUDA_VISIBLE_DEVICES=$2
export NPOP=4
export NREP=2
export NSTEP=2
## case special: t_min / t_max / on(off)_target_scripts / sub_smiles
## need to prepare: init_molecule / pocket_features
export SAVE_DIR=${SCRIPT_DIR}/results
mkdir -p $SAVE_DIR
python -u $SCRIPT_DIR/../../diff_opt_selective.py \
    --params_path $SCRIPT_DIR/../../checkpoints/diffusion_transformer/params.pkl \
    --config_path $SCRIPT_DIR/../../checkpoints/diffusion_transformer/config.pkl \
    --logger_path $SAVE_DIR/opt_bike-mpsk1_$(date "+%Y-%m-%d")_seed${1}_npop${NPOP}_nrep${NREP}_step${NSTEP}/Logs.txt \
    --save_path $SAVE_DIR/opt_bike-mpsk1_$(date "+%Y-%m-%d")_seed${1}_npop${NPOP}_nrep${NREP}_step${NSTEP} \
    --random_seed $1 \
    --np_random_seed $1 \
    --total_step $NSTEP \
    --device_batch_size $NPOP \
    --t_min 15 \
    --t_max 20 \
    --n_replicate $NREP \
    --on_target_scripts $SCRIPT_DIR/dsdp_bike.sh \
    --off_target_scripts $SCRIPT_DIR/dsdp_mpsk1.sh \
    --vae_config_path $SCRIPT_DIR/../../checkpoints/auto_encoder/config.pkl \
    --vae_params_path $SCRIPT_DIR/../../checkpoints/auto_encoder/params.pkl \
    --alphabet_path $SCRIPT_DIR/../../train/smiles_alphabet.pkl \
    --init_molecule_path $SCRIPT_DIR/init_molecule/init_search_molecule.pkl \
    --sub_smiles 'C'
exit