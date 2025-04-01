"""
    Main script for diffusion ES evolution.
    DiffusionES From: https://github.com/bhyang/diffusion-es.
    Note that this code is for single device!
"""

import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
import jax
import logging
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
import pickle as pkl
import argparse
import datetime
import functools

from tqdm import tqdm
from ml_collections import ConfigDict
from src.model.diffusion_transformer import DiffusionTransformer
from src.common.utils import safe_l2_normalize
from train.scheduler import GaussianDiffusion
from train.inference import InferEncoder, InferDecoder, Inferencer, \
    tokens2smiles, smi2graph_features, standardize
from train.rewards import LogP_reward, tanimoto_sim, \
    dsdp_reward, dsdp_batch_reward, QED_reward, SA_reward
from utils import NSGA_II, find_repeats, expand_batch_dim, \
    decoder_function, encoder_function
from rdkit import Chem

from configs import global_config as default_global_config
from configs import dit_config as default_net_config
from configs import train_config as default_train_config

def infer(args):

    #################################################################################
    #               Setting constants, recoder and loading networks                 #
    #################################################################################

    #### set constants
    TOTAL_STEP = args.total_step
    DEVICE_BATCH_SIZE = args.device_batch_size
    N_TOKENS = args.num_latent_tokens
    D_LATENT = args.dim_latent
    N_EQ_STEPS = args.eq_steps
    N_REPLICATE = args.n_replicate
    os.makedirs(args.save_path, exist_ok=True)

    #### set recoder
    recoder = logging.getLogger("inferencing dit")
    recoder.setLevel(level = logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level = logging.DEBUG)
    recoder.addHandler(stream_handler)
    file_handler = logging.FileHandler(args.logger_path)
    file_handler.setLevel(level = logging.DEBUG)
    recoder.addHandler(file_handler)

    #### load config
    if args.config_path:
        with open(args.config_path, 'rb') as f:
            config_dicts = pkl.load(f)
        global_config = ConfigDict(config_dicts['global_config'])
        net_config = ConfigDict(config_dicts['net_config'])
        train_config = ConfigDict(config_dicts['train_config'])
    else:
        global_config = default_global_config
        net_config = default_net_config
        train_config = default_train_config
    DIFFUSION_TIMESTEPS = args.diffusion_timesteps
    global_config.dropout_flag = False
    # vae config
    with open(args.vae_config_path, 'rb') as f:
        config_dicts = pkl.load(f)
    vae_config = ConfigDict(config_dicts['net_config'])
    data_config = ConfigDict(config_dicts['data_config'])
    vae_global_config = ConfigDict(config_dicts['global_config'])
    vae_global_config.dropout_flag = False

    #### load net
    dit_net = DiffusionTransformer(net_config, global_config)
    scheduler = GaussianDiffusion(train_config,)
    encoding_net = InferEncoder(vae_config, vae_global_config)
    decoding_net = InferDecoder(vae_config, vae_global_config)

    #### load params
    with open(args.vae_params_path, 'rb') as f:
        vae_params = pkl.load(f)
        vae_params = jtu.tree_map(lambda x: jnp.asarray(x), vae_params)
    encoder_params = {
        'Encoder_0': vae_params['params']['generator']['Encoder_0'],
        'Dense_0': vae_params['params']['generator']['Dense_0'],
    }
    decoder_params = {
        'Decoder_0': vae_params['params']['generator']['Decoder_0'],
    }
    with open(args.params_path, 'rb') as f:
        params = pkl.load(f)
        params = jtu.tree_map(jnp.asarray, params)
    
    #### set inferencer
    beam_size = args.beam_size
    default_infer_config = {
        'sampling_method': args.sampling_method,
        'device_batch_size': args.device_batch_size,
        'n_seq_length': data_config['n_pad_token'],
        'beam_size': beam_size,
        'bos_token': data_config['bos_token_id'],
        'eos_token': data_config['eos_token_id'],
        'n_local_device': jax.local_device_count(),
        'num_prefix_tokens': data_config['n_query_tokens'],
        'step_limit': 160,
    }
    if args.infer_config_path:
        with open(args.infer_config_path, 'rb') as f:
            infer_config = ConfigDict(pkl.load(f))
    else:
        infer_config = ConfigDict(default_infer_config)
    inferencer = Inferencer(
        encoding_net, decoding_net, encoder_params, decoder_params, infer_config)
    
    #################################################################################
    #                    Defining functions for searching steps                     #
    #################################################################################

    #### define encoder & decoder functions
    with open(args.alphabet_path, 'rb') as f: # load alphabet
        alphabet: dict = pkl.load(f)
        alphabet = alphabet['symbol_to_idx']
    reverse_alphabet = {v: k for k, v in alphabet.items()}
    decoder_f = functools.partial(decoder_function,
        inferencer = inferencer, reverse_alphabet = reverse_alphabet)
    encoder_f = functools.partial(encoder_function, inferencer = inferencer)
    
    ### define reward function
    os.makedirs(os.path.join(args.save_path, 'ligands'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'logs'), exist_ok=True)
    def reward_function(molecule_dict, cached_dict):
        ## for repeat molecules, we use cached scores.

        ## get unique smiles in this iter.
        unique_smiles = cached_dict['unique_smiles']
        unique_scores = cached_dict['unique_scores']
        todo_smiles = molecule_dict['smiles'] # (dbs * r,)
        todo_unique_smiles = np.unique(todo_smiles)
        todo_unique_smiles = np.setdiff1d(todo_unique_smiles, unique_smiles)
        todo_unique_scores = np.empty((0, 3), dtype = np.float32)

        # breakpoint()
        ## we use dsdp docking reward + QED reward
        if todo_unique_smiles.size > 0:
            ## run dsdp
            print('---------------PROT-1 docking---------------')
            r_dock_1 = dsdp_batch_reward(
                smiles = todo_unique_smiles,
                cached_file_path = args.save_path,
                dsdp_script_path = 'cases/case_pi3k/dsdp_pi3k_alpha.sh',
            )
            r_dock_1 = np.asarray(r_dock_1, np.float32) * (-1.) # (N,)
            print('---------------PROT-2 docking---------------')
            r_dock_2 = dsdp_batch_reward(
                smiles = todo_unique_smiles,
                cached_file_path = args.save_path,
                dsdp_script_path = 'cases/case_pi3k/dsdp_pi3k_beta.sh',
                gen_lig_pdbqt = False,
            )
            r_dock_2 = np.asarray(r_dock_2, np.float32) * (-1.) # (N,)
            print('---------------PROT-3 docking---------------')
            r_dock_3 = dsdp_batch_reward(
                smiles = todo_unique_smiles,
                cached_file_path = args.save_path,
                dsdp_script_path = 'cases/case_pi3k/dsdp_pi3k_delta.sh',
                gen_lig_pdbqt = False,
            )
            r_dock_3 = np.asarray(r_dock_3, np.float32) * (-1.)
            r_dock_1_minus_2 = r_dock_1 - r_dock_2 # (N,)
            r_dock_1_minus_3 = r_dock_1 - r_dock_3
            ## optimize for pi3k-a affinity and pi3k-a - pi3k-b affinity
            todo_unique_scores = np.stack([r_dock_1, r_dock_1_minus_2, r_dock_1_minus_3], axis = 1) # (N, 3)
            unique_smiles = np.concatenate([unique_smiles, todo_unique_smiles])
            unique_scores = np.concatenate([unique_scores, todo_unique_scores])

        ## get score for this batch
        todo_index = [np.where(unique_smiles == s)[0][0] for s in todo_smiles]
        todo_scores = unique_scores[todo_index]
        cached_dict['update_unique_smiles'] = todo_unique_smiles
        cached_dict['update_unique_scores'] = todo_unique_scores
        return todo_scores, cached_dict
    
    def constraint_function(molecule_dict, cached, config):

        unique_smiles = cached['unique_smiles']
        ## qed
        qed = np.asarray(QED_reward(molecule_dict['smiles']), np.float32)
        qed_constraint = np.array(qed > config['qed_threshold'], np.int32)
        ## sas
        sas = np.asarray(SA_reward(molecule_dict['smiles']), np.float32)
        sas_constraint = np.array(sas < config['sas_threshold'], np.int32)
        ## LogP
        logp = np.asarray(LogP_reward(molecule_dict), np.float32)
        logp_constraint = np.array((logp >= config['logp_min']) & (logp <= config['logp_max']), np.int32)
        ## test for repeat structure constraint
        rep_constraint = find_repeats(molecule_dict['smiles'], unique_smiles)
        return np.stack([rep_constraint, qed_constraint, logp_constraint, sas_constraint], axis = 1) # (N, 3)
    
    def update_unique(cached):
        unique_smiles = np.concatenate([cached['unique_smiles'], cached['update_unique_smiles']])
        unique_scores = np.concatenate([cached['unique_scores'], cached['update_unique_scores']])
        cached['unique_smiles'] = unique_smiles
        cached['unique_scores'] = unique_scores
        return cached

    #### define noise & denoise functions
    def denoise_step(params, x, mask, time, rope_index, rng_key):
        time = jnp.full((x.shape[0],), time) ## (dbs,)
        eps_pred = dit_net.apply(
            {'params': params['params']['net']}, x, mask, 
            time, tokens_rope_index = rope_index) ## (dbs, npt, d)
        mean, variance, log_variance = scheduler.p_mean_variance(
            x, time, eps_pred, clamp_x0_fn = None, clip = False)
        rng_key, sub_key = jax.random.split(rng_key)
        x = mean + jnp.exp(0.5 * log_variance) * jax.random.normal(sub_key, x.shape)
        return x, rng_key
    jit_denoise_step = jax.jit(denoise_step)
    
    def noise_step(x, time, rng_key):
        time = jnp.full((x.shape[0],), time) ## (dbs,)
        rng_key, sub_key = jax.random.split(rng_key)
        x = scheduler.q_sample_step(x, time, jax.random.normal(sub_key, x.shape))
        return x, rng_key
    jit_noise_step = jax.jit(noise_step)

    def noise(x, time, rng_key):
        """q(x_t | x_0)"""
        time = jnp.full((x.shape[0],), time) ## (dbs,)
        rng_key, sub_key = jax.random.split(rng_key)
        x = scheduler.q_sample(x, time, jax.random.normal(sub_key, x.shape))
        return x, rng_key
    jit_noise = jax.jit(noise)

    def replicate_func(x):
        ### (dbs, ...) -> (dbs, r, ...) -> (dbs * r, ...)
        repeat_x = np.repeat(x[:, None], N_REPLICATE, axis = 1)
        return repeat_x.reshape(-1, *repeat_x.shape[2:])

    #################################################################################
    #                            Defining main functions                            #
    #################################################################################

    def diffusion_es_search_step(step_it, x, rng_key, config, cached):
        ### x: (dbs * r, npt, d)

        mask_x = cached['mask'] ## (dbs * r, npt)
        rope_index_x = cached['rope_index'] ## (dbs * r, npt)
        cached_smiles = [d['smiles'] for d in cached['molecules']] ## (dbs,)
        diffusion_time_it = config.time[step_it]

        ### decoding to molecules: {'graphs', 'smiles',}, (dbs * r, ...)
        decode_molecules = decoder_f(
            x, replicate_func(cached_smiles[-1]))
        # breakpoint() ## check here

        ### scoring: (dbs * r,)
        scores, cached = reward_function(decode_molecules, cached) # (dbs * r, m)
        # breakpoint() ## check here
        constraints = constraint_function(decode_molecules, cached, config,)
        # breakpoint() ## check here
        cached = update_unique(cached,)

        ### concat father populations
        scores = np.concatenate(
            [cached['scores'][-1], scores], axis = 0) # (dbs * r + dbs, m)
        constraints = np.concatenate(
            [cached['constraints'][-1], constraints], axis = 0) # (dbs * r + dbs, c)
        decode_molecules = jtu.tree_map(
            lambda x, y: np.concatenate([x, y], axis = 0), 
            cached['molecules'][-1], decode_molecules)
        
        ### choicing using NSGA-II
        # breakpoint() ## check here
        choiced_idx = NSGA_II(scores, constraints, 
                              config.constraint_weights, n_pops = DEVICE_BATCH_SIZE)

        ### sampling: (dbs,)
        choiced_molecules = jtu.tree_map(
            lambda x: x[choiced_idx], decode_molecules) ## (dbs, ...)
        choiced_scores = scores[choiced_idx] ## (dbs,)
        choiced_constraints = constraints[choiced_idx]
        recoder.info(
            f'Top 4 DSDP PI3K-A scores: {np.round(np.sort(choiced_scores[:, 0])[-4:], decimals=3)}')
        recoder.info(
            f'Top 4 DSDP PI3K-A - PI3K-B scores: {np.round(np.sort(choiced_scores[:, 1])[-4:], decimals=3)}')
        recoder.info(
            f'Top 4 DSDP PI3K-A - PI3K-D scores: {np.round(np.sort(choiced_scores[:, 2])[-4:], decimals=3)}')
        # breakpoint() ## check here

        ### save
        cached['molecules'].append(choiced_molecules)
        cached['scores'].append(choiced_scores)
        cached['constraints'].append(choiced_constraints)
        # breakpoint() ## check here

        ### encoding: (dbs, npt, d) -> (dbs * r, npt, d)
        choiced_x = encoder_f(choiced_molecules['graphs'])
        choiced_x = replicate_func(choiced_x)
        choiced_x *= jnp.sqrt(choiced_x.shape[-1]) ## scale here
        # breakpoint() ## check here

        ### renoise & denoise
        x_out, rng_key = jit_noise(choiced_x, diffusion_time_it, rng_key)
        for t_i in tqdm(range(diffusion_time_it)):
            t = diffusion_time_it - t_i
            ## we run some eq steps first for efficient sampling
            for eq_step in range(config.eq_steps):
                x_out, rng_key = jit_denoise_step(params, x_out, mask_x, t, rope_index_x, rng_key)
                x_out, rng_key = jit_noise_step(x_out, t, rng_key)
            ## x: (dbs *  r, npt, d)
            x_out, rng_key = jit_denoise_step(params, x_out, mask_x, t, rope_index_x, rng_key)
        return x_out, cached, rng_key

    #### define diffusion es main function
    def diffusion_es(config, rng_key, init_molecules):

        ### init cached molecules
        init_scores = init_molecules['scores']
        assert init_scores.ndim == 2, f'{init_scores.ndim} != 2'
        init_constraints = np.stack([
            np.array([1,] + [0 for _ in range(init_scores.shape[0] - 1)], np.int32), # rep
            np.ones((init_scores.shape[0],), np.int32), # qed
            np.ones((init_scores.shape[0],), np.int32), # logp
            np.ones((init_scores.shape[0],), np.int32), # sas
        ], axis = 1)

        ### prepare
        init_key, rng_key = jax.random.split(rng_key)
        x = jax.random.normal(
            init_key, (DEVICE_BATCH_SIZE, N_TOKENS, D_LATENT))
        x = replicate_func(x) ## (dbs * r, npt, dim)
        m = jnp.ones(
            (DEVICE_BATCH_SIZE * N_REPLICATE, N_TOKENS), jnp.int32)
        rope_index = jnp.array(
            [np.arange(N_TOKENS),] * (DEVICE_BATCH_SIZE * N_REPLICATE), 
            dtype = jnp.int32).reshape(DEVICE_BATCH_SIZE * N_REPLICATE, N_TOKENS)
        
        ### the first offsprings
        recoder.info(f'Generating init offsprings...')
        for t_i in tqdm(range(DIFFUSION_TIMESTEPS)):
            t = DIFFUSION_TIMESTEPS - t_i
            ### we run some eq steps first for efficient sampling
            for eq_step in range(N_EQ_STEPS):
                x, rng_key = jit_denoise_step(params, x, m, t, rope_index, rng_key)
                x, rng_key = jit_noise_step(x, t, rng_key)
            ### x: (n_device, dbs, npt, d)
            x, rng_key = jit_denoise_step(params, x, m, t, rope_index, rng_key)
        
        ### search steps
        cached = {
            'mask': m, 
            'rope_index': rope_index,
            'molecules': [{'smiles': init_molecules['smiles'], 'graphs': init_molecules['graphs']},], 
            'scores': [init_scores], 
            'constraints': [init_constraints], 
            'unique_smiles': init_molecules['smiles'][:1], 
            'unique_scores': init_scores[:1],
            }
        recoder.info(f'Starting search, total steps = {config.search_steps}')
        for step in range(config.search_steps):
            recoder.info(f'----------------------------------------------------------------')
            recoder.info(f'Searching step {step + 1}, noise mutation steps {config.time[step]}')
            x, cached, rng_key = diffusion_es_search_step(step, x, rng_key, config, cached)
        recoder.info(f'----------------------------------------------------------------')
        
        ### decode & evaluate
        decode_molecules = decoder_f(
            x, replicate_func(cached['molecules'][-1]['smiles']))
        scores, cached = reward_function(decode_molecules, cached)
        constraints = constraint_function(decode_molecules, cached, config)
        cached = update_unique(cached,)
        ### concat father populations
        scores = np.concatenate(
            [cached['scores'][-1], scores], axis = 0) # (dbs * r + dbs, m)
        constraints = np.concatenate(
            [cached['constraints'][-1], constraints], axis = 0) # (dbs * r + dbs, c)
        decode_molecules = jtu.tree_map(
            lambda x, y: np.concatenate([x, y], axis = 0), 
            cached['molecules'][-1], decode_molecules)

        ### search last step
        choiced_idx = NSGA_II(scores, constraints, 
                              config.constraint_weights, n_pops = DEVICE_BATCH_SIZE)
        choiced_molecules = jtu.tree_map(
            lambda x: x[choiced_idx], decode_molecules) ## (dbs, ...)
        choiced_scores = scores[choiced_idx] ## (dbs,)
        choiced_constraints = constraints[choiced_idx]
        ### save
        cached['molecules'].append(choiced_molecules) ## (dbs, ...)
        cached['scores'].append(choiced_scores)
        cached['constraints'].append(choiced_constraints)
        return choiced_molecules, cached

    #################################################################################
    #                          Executing searching steps                            #
    #################################################################################

    #### load params
    rng_key = jax.random.PRNGKey(args.random_seed)
    np.random.seed(args.np_random_seed)

    #### recoding info
    args_dict = vars(args)
    recoder.info(f'=====================INPUT ARGS=====================')
    recoder.info("INPUT ARGS:")
    for k, v in args_dict.items():
        recoder.info(f"\t{k}: {v}")
    
    #### inference
    with open(args.init_molecule_path, 'rb') as f:
        lead_molecules = pkl.load(f)
        lead_molecules = expand_batch_dim(DEVICE_BATCH_SIZE, lead_molecules)
    print(jtu.tree_map(lambda x: x.shape, lead_molecules))
    time_sched = np.array([500, 150, 100] * (TOTAL_STEP // 3 + 1), np.int32)[:TOTAL_STEP]
    # breakpoint() ## check here
    search_config = ConfigDict({
        'time': time_sched,
        'eq_steps': N_EQ_STEPS,
        'search_steps': TOTAL_STEP,
        'qed_threshold': 0.5,
        'sas_threshold': 4.5,
        'logp_min': 2.0,
        'logp_max': 6.0,
        'constraint_weights': None})
    infer_start_time = datetime.datetime.now()
    recoder.info(f'=====================START INFERENCE=====================')
    output_molecules, cached = diffusion_es(search_config, rng_key, lead_molecules)
    ## save
    save_file = {
        'smiles': [c['smiles'] for c in cached['molecules']],
        'scores': cached['scores'],
        'constraints': cached['constraints'],
    }
    save_path = os.path.join(args.save_path, f'diffusion_es_opt.pkl')
    with open(save_path, 'wb') as f:
        pkl.dump(save_file, f)
    
    ## inference done
    recoder.info(f'=====================END INFERENCE=====================')
    tot_time = datetime.datetime.now() - infer_start_time
    recoder.info(f'Inference done, time {tot_time}, results saved to {args.save_path}')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type = str, default = None)
    parser.add_argument('--params_path', type = str, required = True)
    parser.add_argument('--logger_path', type = str, required = True)
    parser.add_argument('--save_path', type = str, required = True)
    parser.add_argument('--random_seed', type = int, default = 42)
    parser.add_argument('--np_random_seed', type = int, default = 42)
    parser.add_argument('--total_step', type = int, required = True)
    parser.add_argument('--device_batch_size', type = int, required = True)
    parser.add_argument('--num_latent_tokens', type = int, required = True)
    parser.add_argument('--dim_latent', type = int, required = True)
    parser.add_argument('--eq_steps', type = int, default = 10)
    parser.add_argument('--callback_step', type = int, default = 10)
    parser.add_argument('--beam_size', type = int, default = 5)
    parser.add_argument('--sampling_method', type = str, default = 'beam')
    parser.add_argument('--infer_config_path', type = str, default = None)
    parser.add_argument('--vae_config_path', type = str, required = True)
    parser.add_argument('--vae_params_path', type = str, required = True)
    parser.add_argument('--alphabet_path', type = str, required = True)
    parser.add_argument('--n_replicate', type = int, default = 1)
    parser.add_argument('--init_molecule_path', type = str, required = True)
    parser.add_argument('--diffusion_timesteps', type = int, default = 500)
    args = parser.parse_args()

    infer(args)