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
python -u $SCRIPT_DIR/../../diff_opt_base.py \
    --params_path $SCRIPT_DIR/../../checkpoints/diffusion_transformer/params.pkl \
    --config_path $SCRIPT_DIR/../../checkpoints/diffusion_transformer/config.pkl \
    --logger_path $SAVE_DIR/opt_jnk3-gsk3b_$(date "+%Y-%m-%d")_seed${1}_npop${NPOP}_nrep${NREP}_step${NSTEP}/Logs.txt \
    --save_path $SAVE_DIR/opt_jnk3-gsk3b_$(date "+%Y-%m-%d")_seed${1}_npop${NPOP}_nrep${NREP}_step${NSTEP} \
    --random_seed $1 \
    --np_random_seed $1 \
    --total_step $NSTEP \
    --device_batch_size $NPOP \
    --t_min 110 \
    --t_max 135 \
    --use_prompt \
    --n_replicate $NREP \
    --on_target_scripts $SCRIPT_DIR/dsdp_jnk3.sh $SCRIPT_DIR/dsdp_gsk3b.sh \
    --vae_config_path $SCRIPT_DIR/../../checkpoints/auto_encoder/config.pkl \
    --vae_params_path $SCRIPT_DIR/../../checkpoints/auto_encoder/params.pkl \
    --alphabet_path $SCRIPT_DIR/../../train/smiles_alphabet.pkl \
    --init_molecule_path $SCRIPT_DIR/init_molecule/init_search_molecule.pkl \
    --sub_smiles 'O=C(N1CCNCC1)c1ccccc1' \
    --pocket_features_path $SCRIPT_DIR/pocket_features/pocket_features.pkl \
    --affinity_predictor_params_path $SCRIPT_DIR/../../checkpoints/affinity_predictor/params.pkl \
    --affinity_predictor_config_path $SCRIPT_DIR/../../checkpoints/affinity_predictor/config.pkl
exit