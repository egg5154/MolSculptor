"""
    Main script for diffusion-evolution optimization.
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
from train.inference import InferEncoder, InferDecoder, Inferencer
from train.rewards import QED_reward, SA_reward
from utils import NSGA_II, expand_batch_dim, encoder_function, decoder_function, \
    dual_inhibitor_reward_function, has_substructure, find_repeats, sim_function

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
    encoder_f = functools.partial(encoder_function, inferencer = inferencer)
    decoder_f = functools.partial(
        decoder_function, inferencer = inferencer, reverse_alphabet = reverse_alphabet, beam_size = beam_size)

    #### make cache dirs for DSDP
    os.makedirs(os.path.join(args.save_path, 'ligands'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'outputs'), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, 'logs'), exist_ok=True)

    #### define reward functions
    def reward_function(molecule_dict, cached):
        return dual_inhibitor_reward_function(molecule_dict, cached,
            [args.dsdp_script_path_1, args.dsdp_script_path_2], args.save_path,)
    
    def constraint_function(molecule_dict, cached, config):

        template_smiles = replicate_func(cached['molecules'][0]['smiles'])
        unique_smiles = cached['unique_smiles']

        sim = sim_function(molecule_dict['smiles'], template_smiles)
        sim_constraint = np.array(sim > config['sim_threshold'], np.int32) # (N,)
        qed = np.asarray(QED_reward(molecule_dict['smiles']), np.float32)
        qed_constraint = np.array(qed > config['qed_threshold'], np.int32)
        sas = np.asarray(SA_reward(molecule_dict['smiles']), np.float32)
        sas_constraint = np.array(sas < config['sas_threshold'], np.int32)
        sub_constraint = np.asarray(has_substructure(molecule_dict['smiles'], args.sub_smiles), np.int32) # (N,)
        rep_constraint = find_repeats(molecule_dict['smiles'], unique_smiles) # to avoid duplicates
        return np.stack([sub_constraint, rep_constraint, sim_constraint, 
                         qed_constraint, sas_constraint], axis = 1) # (N, 5)
    
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
        decode_molecules = decoder_f(x, replicate_func(cached_smiles[-1]))
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
        choiced_sim_scores = sim_function(
            choiced_molecules['smiles'], cached_smiles[0])
        recoder.info(f'Top 4 DSDP PROT-1 scores: {np.round(np.sort(choiced_scores[:, 0])[-4:], decimals=3)}')
        recoder.info(f'Top 4 DSDP PROT-2 scores: {np.round(np.sort(choiced_scores[:, 1])[-4:], decimals=3)}')
        recoder.info(f'Average sim score: {np.mean(choiced_sim_scores):.3f}')

        ### save
        cached['molecules'].append(choiced_molecules)
        cached['scores'].append(choiced_scores)
        cached['sim'].append(choiced_sim_scores)
        cached['constraints'].append(choiced_constraints)

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
    def diffusion_es_optimization(config, rng_key, init_molecules,):
        ### init_molecules: dict {'graphs', 'smiles',}, has batch dim.

        ### evalutating
        init_scores = init_molecules['scores']
        assert init_scores.ndim == 2, f'{init_scores.ndim} != 2'
        init_sim_scores = np.ones_like(init_scores) # (dbs,)
        # init_constraints = np.ones((init_scores.shape[0], 5), np.int32) ## WARNING: should same with #constraints
        init_constraints = np.stack([
            np.ones((init_scores.shape[0],), np.int32), # sub
            np.array([1,] + [0 for _ in range(init_scores.shape[0] - 1)], np.int32), # rep
            np.ones((init_scores.shape[0],), np.int32), # sim
            np.ones((init_scores.shape[0],), np.int32), # qed
            np.ones((init_scores.shape[0],), np.int32), # sas
        ], axis = 1)

        ### prepare
        init_key, rng_key = jax.random.split(rng_key)
        x = encoder_f(init_molecules['graphs']) ## (dbs, npt, dim)
        x = x * jnp.sqrt(x.shape[-1]) ## scale here
        x = replicate_func(x) ## (dbs * r, npt, dim)
        m = jnp.ones(
            (DEVICE_BATCH_SIZE * N_REPLICATE, N_TOKENS), jnp.int32)
        rope_index = \
            jnp.array(
                [np.arange(N_TOKENS),] * (DEVICE_BATCH_SIZE * N_REPLICATE), 
                dtype = jnp.int32).reshape(DEVICE_BATCH_SIZE * N_REPLICATE, N_TOKENS)
        
        ### the first offsprings
        recoder.info(f'Generating init offsprings...')
        init_t = (args.t_min + args.t_max) // 2
        x, rng_key = jit_noise(x, init_t, init_key)
        for t_i in tqdm(range(init_t)):
            t = init_t - t_i
            ### we run some eq steps first for efficient sampling
            for eq_step in range(N_EQ_STEPS):
                x, rng_key = jit_denoise_step(params, x, m, t, rope_index, rng_key)
                x, rng_key = jit_noise_step(x, t, rng_key)
            ### x: (n_device, dbs, npt, d)
            x, rng_key = jit_denoise_step(params, x, m, t, rope_index, rng_key)
        
        ### search steps
        cached = {
            'mask': m, 'rope_index': rope_index,
            'molecules': [{'smiles': init_molecules['smiles'], 'graphs': init_molecules['graphs']},], 
            'scores': [init_scores], 'sim': [init_sim_scores],
            'constraints': [init_constraints], 
            'unique_smiles': init_molecules['smiles'][:1], 'unique_scores': init_scores[:1],}
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
        sim_scores = np.asarray(sim_function(
            decode_molecules['smiles'], replicate_func(cached['molecules'][0]['smiles'])), np.float32)
        constraints = constraint_function(decode_molecules, cached, config)
        cached = update_unique(cached,)
        ### concat father populations
        scores = np.concatenate(
            [cached['scores'][-1], scores], axis = 0) # (dbs * r + dbs, m)
        sim_scores = np.concatenate(
            [cached['sim'][-1], sim_scores], axis = 0)
        constraints = np.concatenate(
            [cached['constraints'][-1], constraints], axis = 0) # (dbs * r + dbs, c)
        decode_molecules = jtu.tree_map(
            lambda x, y: np.concatenate([x, y], axis = 0), 
            cached['molecules'][-1], decode_molecules)

        ### final population
        choiced_idx = NSGA_II(scores, constraints, 
                              config.constraint_weights, n_pops = DEVICE_BATCH_SIZE)
        choiced_molecules = jtu.tree_map(
            lambda x: x[choiced_idx], decode_molecules) ## (dbs, ...)
        choiced_scores = scores[choiced_idx] ## (dbs,)
        choiced_constraints = constraints[choiced_idx]
        choiced_sim_scores = sim_scores[choiced_idx]
        ### save
        cached['molecules'].append(choiced_molecules) ## (dbs, ...)
        cached['scores'].append(choiced_scores)
        cached['sim'].append(choiced_sim_scores)
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
    time_sched = np.random.randint(args.t_min, args.t_max, size = (TOTAL_STEP,))
    search_config = ConfigDict({
        'time': time_sched, 'eq_steps': N_EQ_STEPS,
        'search_steps': TOTAL_STEP,
        'constraint_weights': None,
        'sim_threshold': 0.4,
        'qed_threshold': 0.5,
        'sas_threshold': 6.0,})
    infer_start_time = datetime.datetime.now()
    recoder.info(f'=====================START INFERENCE=====================')
    output_molecules, cached = diffusion_es_optimization(search_config, rng_key, lead_molecules)
    ## save
    save_file = {
        'smiles': [c['smiles'] for c in cached['molecules']],
        'scores': cached['scores'],
        'sim': cached['sim'],
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
    parser.add_argument('--beam_size', type = int, default = 4)
    parser.add_argument('--sampling_method', type = str, default = 'beam')
    parser.add_argument('--infer_config_path', type = str, default = None)
    parser.add_argument('--vae_config_path', type = str, required = True)
    parser.add_argument('--vae_params_path', type = str, required = True)
    parser.add_argument('--alphabet_path', type = str, required = True)
    parser.add_argument('--n_replicate', type = int, default = 1)
    parser.add_argument('--init_molecule_path', type = str, required = True)
    parser.add_argument('--dsdp_script_path_1', type = str, required = True)
    parser.add_argument('--dsdp_script_path_2', type = str, required = True)
    parser.add_argument('--t_min', type = int, required = True)
    parser.add_argument('--t_max', type = int, required = True)
    parser.add_argument('--sub_smiles', type = str, required = True)
    args = parser.parse_args()

    infer(args)