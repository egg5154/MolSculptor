"""
    Main script for diffusion-evolution optimization.
"""

import os
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
import jax
import shutil
import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
import pickle as pkl
import flax.linen as nn
import argparse
import datetime
import functools
import math
import optax

from tqdm import tqdm
from logging import getLogger, Logger
from ml_collections import ConfigDict
from src.model.diffusion_transformer import DiffusionTransformer
from src.common.utils import safe_l2_normalize
from train.scheduler import GaussianDiffusion
from train.inference import InferEncoder, InferDecoder, Inferencer
from train.rewards import QED_reward, SA_reward, dsdp_batch_reward
from utils import NSGA_II, expand_batch_dim, encoder_function, decoder_function, \
    has_substructure, find_repeats, sim_function, set_recoder, load_infer_config, \
    load_infer_params, set_infer_config, set_tfg_config
from utils import DockingSurrogateModel, DiffusionTFG
from net.affinity_predictor import ProjectorWithPocket, EncoderWithPocket, Projector, \
    set_projector_config, set_global_config


ITER = 1
DELETE_CACHED_FILES = True
def infer(args):

    # breakpoint()

    #################################################################################
    #               Setting constants, recoder and loading networks                 #
    #################################################################################

    #### set constants
    TOTAL_STEP = args.total_step
    DEVICE_BATCH_SIZE = args.device_batch_size
    N_TOKENS = args.num_latent_tokens
    N_EQ_STEPS = args.eq_steps
    N_REPLICATE = args.n_replicate
    DIM = args.dim_latent
    os.makedirs(args.save_path, exist_ok=True)

    rng_key = jax.random.PRNGKey(args.random_seed)
    np.random.seed(args.np_random_seed)

    recoder = set_recoder(args.logger_path,)

    #### load net
    net_config, train_config, _, global_config = load_infer_config(args.config_path)
    vae_config, _, data_config, vae_global_config = load_infer_config(args.vae_config_path)
    dit_net = DiffusionTransformer(net_config, global_config)
    scheduler = GaussianDiffusion(train_config,)
    encoding_net = InferEncoder(vae_config, vae_global_config)
    decoding_net = InferDecoder(vae_config, vae_global_config)

    #### load params
    params, encoder_params, decoder_params = load_infer_params(
        args.params_path, args.vae_params_path,)
    
    #### set inferencer
    beam_size = args.beam_size
    infer_config = set_infer_config(args, data_config,)
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
    encoder_f = functools.partial(
        encoder_function, inferencer = inferencer)
    decoder_f = functools.partial(
        decoder_function, inferencer = inferencer, reverse_alphabet = reverse_alphabet, beam_size = beam_size)

    #### define reward functions
    n_on_tags = len(args.on_target_scripts)
    if args.off_target_scripts is None:
        args.off_target_scripts = []
    n_off_tags = len(args.off_target_scripts)
    def reward_function(molecule_dict, cached_dict, save_path=args.save_path):
        global ITER
        ## for repeat molecules, we use cached scores.

        """unique_*: cached unique items (graphs, smiles, scores & latents), to avoid recomputing.
        todo_unique_*: new unique items in this iter.
        
        """
        unique_smiles = cached_dict['unique_smiles']
        unique_scores = cached_dict['unique_scores']
        todo_smiles = molecule_dict['smiles'] # (dbs * r,)
        todo_latents = molecule_dict['latents'] # (dbs * r, npt, d)
        todo_unique_smiles, todo_unique_index = np.unique(todo_smiles, return_index=True)
        todo_unique_latents = todo_latents[todo_unique_index]
        is_in_unique_smiles = np.in1d(todo_unique_smiles, unique_smiles)
        tmp_mask = np.logical_not(is_in_unique_smiles)
        todo_unique_smiles = todo_unique_smiles[tmp_mask]
        todo_unique_latents = todo_unique_latents[tmp_mask]
        todo_unique_scores = np.empty((0, n_on_tags + n_off_tags), dtype=np.float32)

        os.makedirs(os.path.join(save_path, f'iter_{ITER}', 'ligands'), exist_ok=True)
        os.makedirs(os.path.join(save_path, f'iter_{ITER}', 'outputs'), exist_ok=True)
        os.makedirs(os.path.join(save_path, f'iter_{ITER}', 'logs'), exist_ok=True)
        if todo_unique_smiles.size > 0:
            ## run dsdp
            todo_unique_scores = []
            gen_pdbqt = True
            for script in args.on_target_scripts:
                r_dock = dsdp_batch_reward(
                    smiles = todo_unique_smiles,
                    cached_file_path = os.path.join(save_path, f'iter_{ITER}'),
                    dsdp_script_path = script,
                    gen_lig_pdbqt = gen_pdbqt)
                gen_pdbqt = False
                r_dock = np.asarray(r_dock, np.float32) * (-1.) # (N,), higher is better
                todo_unique_scores.append(r_dock)
            for script in args.off_target_scripts:
                r_dock = dsdp_batch_reward(
                    smiles = todo_unique_smiles,
                    cached_file_path = os.path.join(save_path, f'iter_{ITER}'),
                    dsdp_script_path = script,
                    gen_lig_pdbqt = False)
                r_dock = np.asarray(r_dock, np.float32) # lower is better
                todo_unique_scores.append(r_dock)
            
            todo_unique_scores = np.stack(todo_unique_scores, axis = -1) # (N, n_on_tags + n_off_tags)
            todo_unique_scores = np.round(todo_unique_scores, decimals=2)
            unique_smiles = np.concatenate([unique_smiles, todo_unique_smiles])
            unique_scores = np.concatenate([unique_scores, todo_unique_scores])
        

        ## get score for this batch
        todo_index = [np.where(unique_smiles == s)[0][0] for s in todo_smiles]
        todo_scores = unique_scores[todo_index]
        cached_dict['update_unique_smiles'] = todo_unique_smiles
        cached_dict['update_unique_scores'] = todo_unique_scores
        cached_dict['update_unique_latents'] = todo_unique_latents
        ITER += 1
        return todo_scores, cached_dict
    
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
        unique_latent = np.concatenate([cached['unique_latents'], cached['update_unique_latents']])
        cached['unique_smiles'] = unique_smiles
        cached['unique_scores'] = unique_scores
        cached['unique_latents'] = unique_latent
        return cached

    #################################################################################
    #                             TFG & Surrogate Model                             #
    #################################################################################

    if args.pocket_features_path is not None:
        assert args.affinity_predictor_config_path is not None
        assert args.affinity_predictor_params_path is not None
        # set affinity predictor
        with open(args.affinity_predictor_config_path, 'rb') as f:
            config_dict = pkl.load(f)
            predictor_global_config = ConfigDict(config_dict['global_config'])
            predictor_encoder_config = ConfigDict(config_dict['encoder_config'])
            predictor_projector_config = ConfigDict(config_dict['projector_config'])
        
        # load pocket features
        with open(args.pocket_features_path, 'rb') as f:
            pocket_features = pkl.load(f)
            n_pockets = pocket_features['residue_mask'].shape[0]
        
        # load pretrained params
        with open(args.affinity_predictor_params_path, 'rb') as f:
            pretrained_params = pkl.load(f)
            projector_pretrained_params = pretrained_params['params'].pop('out_projector')
            projector_params = {'params': {}}
            for i in range(n_pockets):
                projector_params['params'][f'out_projector_{i}'] = projector_pretrained_params
            encoder_params = pretrained_params
        
        projector = ProjectorWithPocket(
            predictor_projector_config, predictor_global_config)
        encoder = EncoderWithPocket(
            predictor_encoder_config, predictor_global_config, pocket_features)
    else:
        encoder, encoder_params = None, None
        projector_params = None
        projector_config = set_projector_config(DIM, 9.0, 2.5, 0.1, n_on_tags + n_off_tags,)
        projector = Projector(projector_config, set_global_config())
    surrogate_model_config, tfg_config = set_tfg_config(
        score_weights = jnp.asarray([-1.] * n_on_tags + [1.] * n_off_tags),
        eq_steps = args.eq_steps,
        npt = N_TOKENS, dim = DIM,)

    optimizer = optax.adamw(
        # learning_rate = surrogate_model_config.train.learning_rate,
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value = 2e-4, peak_value = 1e-3, end_value = 1e-4,
            warmup_steps = 1000, decay_steps = 40000,
        ),
        weight_decay = surrogate_model_config.train.weight_decay,
    )

    surrogate_model = DockingSurrogateModel(
        config = surrogate_model_config,
        projector_net = projector,
        optimizer = optimizer,
        recoder = recoder,
        encoder_net = encoder,
        encoder_params = encoder_params,
    )
    
    if projector_params is None:
        projector_params = surrogate_model.init_net(rng_key, N_TOKENS, DIM)
    predictor_opt_state = surrogate_model.init_optimizer(projector_params)

    loss_fn = surrogate_model.tfg_loss_fn
    diffusion_guidance = DiffusionTFG(
        config = tfg_config,
        dit_net = dit_net, dit_params = params,
        scheduler = scheduler, loss_fn = loss_fn,
    )
    jit_noise = diffusion_guidance.jit_noise
    jit_noise_step = diffusion_guidance.jit_noise_step
    jit_denoise = diffusion_guidance.jit_denoise
    jit_denoise_step = diffusion_guidance.jit_denoise_step

    def replicate_func(x):
        # (dbs, ...) -> (dbs, r, ...) -> (dbs * r, ...)
        repeat_x = np.repeat(x[:, None], N_REPLICATE, axis = 1)
        return repeat_x.reshape(-1, *repeat_x.shape[2:])

    #################################################################################
    #                                 Search step                                   #
    #################################################################################

    def diffusion_es_search_step(step_it, x, rng_key, config, cached):
        ### x: (dbs * r, npt, d)
        global ITER

        cached_smiles = [d['smiles'] for d in cached['molecules']] # (dbs,)
        cached_latent_tokens = [d['latents'] for d in cached['molecules']] # (dbs, npt, d)
        diffusion_time_it = config.time[step_it]

        """Decoding to molecules: {'graphs', 'smiles', 'latents',} 
        Shape of (dbs * r, ...)
        """
        decode_molecules = decoder_f(
            x, replicate_func(cached_smiles[-1]), replicate_func(cached_latent_tokens[-1]),
            )

        ### scoring: (dbs * r,)
        scores, cached = reward_function(decode_molecules, cached) # (dbs * r, m)
        constraints = constraint_function(decode_molecules, cached, config,)
        cached = update_unique(cached,)

        """Training surrogate model.
        We predict positive scores, and in TFG weights are set for on-targets and off-targets differently.
        NOTE: before training, we need to scale the latents here.
        """
        unique_latents = cached['unique_latents'] # (N, npt, d)
        unique_scores = cached['unique_scores'] # (N, m), on-targets: mainly positive, off-targets: mainly negative
        surrogate_model.add_data(
            x = unique_latents * jnp.sqrt(unique_latents.shape[-1]),
            # higher is better for all scores (because of same scale-shift)
            y = unique_scores * jnp.asarray([1.,] * n_on_tags + [-1.,] * n_off_tags)[None, :] # jnp.abs(unique_scores),
        )
        _params = cached['predictor_params']
        _opt_state = cached['predictor_opt_state']
        _save_path = os.path.join(args.save_path, f'iter_{ITER}', 'predictor')
        os.makedirs(_save_path, exist_ok=True)
        rng_key, _params, _opt_state = surrogate_model.train(
            rng_key, _params, _opt_state, _save_path,)
        cached['predictor_params'] = _params
        cached['predictor_opt_state'] = _opt_state

        # delete cached files
        if DELETE_CACHED_FILES:
            shutil.rmtree(os.path.join(args.save_path, f'iter_{ITER - 1}',))

        """Evolution Algorithm step.
        We combined the parent population and offsprings.
        """
        scores = np.concatenate(
            [cached['scores'][-1], scores], axis = 0) # (dbs * r + dbs, m)
        constraints = np.concatenate(
            [cached['constraints'][-1], constraints], axis = 0) # (dbs * r + dbs, c)
        decode_molecules = jtu.tree_map(
            lambda x, y: np.concatenate([x, y], axis = 0),
            cached['molecules'][-1], decode_molecules)
        choiced_idx = NSGA_II(scores, constraints, 
                              config.constraint_weights, n_pops = DEVICE_BATCH_SIZE)
        choiced_molecules = jtu.tree_map(
            lambda x: x[choiced_idx], decode_molecules) ## (dbs, ...)
        choiced_scores = scores[choiced_idx] ## (dbs,)
        choiced_constraints = constraints[choiced_idx]
        choiced_sim_scores = sim_function(
            choiced_molecules['smiles'], cached_smiles[0])
        for _i in range(choiced_scores.shape[-1]):
            recoder.info(f'Top scores {_i}: {np.sort(choiced_scores[:, _i])[-4:]}')
        recoder.info(f'Average sim score: {np.mean(choiced_sim_scores):.3f}')

        """Save here.
        """
        cached['molecules'].append(choiced_molecules)
        cached['scores'].append(choiced_scores)
        cached['sim'].append(choiced_sim_scores)
        cached['constraints'].append(choiced_constraints)
        # breakpoint() # debug

        """Get latents for next generation.
        Renoise and denoise with TFG.
        """
        ### encoding: (dbs, npt, d) -> (dbs * r, npt, d)
        choiced_x = encoder_f(choiced_molecules['graphs'])
        choiced_x = replicate_func(choiced_x)
        choiced_x *= jnp.sqrt(choiced_x.shape[-1]) # scale here

        ### renoise & denoise
        # breakpoint()
        x_out, rng_key = jit_noise(choiced_x, diffusion_time_it, rng_key)
        x_out, rng_key = diffusion_guidance.tfg_denoise(
            x_out, _params, diffusion_time_it, step_it, cached['mask'], cached['rope_index'], 
            rng_key, cached['prompt_labels'], cached['force_drop_ids'])
        return x_out, cached, rng_key

    #################################################################################
    #                            Defining main function                             #
    #################################################################################

    def diffusion_es_optimization(config, rng_key, init_molecules,):
        ### init_molecules: dict {'graphs', 'smiles',}, has batch dim.

        """Create initial scores, similarity & constraints.
        Note that we add repeat constraints here to avoid duplicates.
        """
        init_scores = init_molecules['scores']
        init_sim_scores = np.ones_like(init_scores) # (dbs,)
        init_constraints = np.stack([
            np.ones((init_scores.shape[0],), np.int32), # substructure
            np.array([1,] + [0 for _ in range(init_scores.shape[0] - 1)], np.int32), # rep
            np.ones((init_scores.shape[0],), np.int32), # sim
            np.ones((init_scores.shape[0],), np.int32), # qed
            np.ones((init_scores.shape[0],), np.int32), # sas
        ], axis = 1)

        ### prepare
        init_key, rng_key = jax.random.split(rng_key)
        x = encoder_f(init_molecules['graphs']) ## (dbs, npt, dim)
        init_molecules['latents'] = np.array(x, np.float32) # save latents
        x = x * jnp.sqrt(x.shape[-1]) # scale here
        x = replicate_func(x) ## (dbs * r, npt, dim)
        mask = jnp.ones(
            (DEVICE_BATCH_SIZE * N_REPLICATE, N_TOKENS), jnp.int32)
        rope_index = \
            jnp.array(
                [np.arange(N_TOKENS),] * (DEVICE_BATCH_SIZE * N_REPLICATE), 
                dtype = jnp.int32).reshape(DEVICE_BATCH_SIZE * N_REPLICATE, N_TOKENS)
        
        ### the first offsprings
        recoder.info(f'Generating init offsprings...')
        init_t = (args.t_min + args.t_max) // 2
        x, rng_key = jit_noise(x, init_t, init_key)
        prompt_labels, force_drop_ids = diffusion_guidance.generate_input_labels(
            DEVICE_BATCH_SIZE, N_REPLICATE, 
            {'qed_min': config.qed_threshold, 'sas_max': config.sas_threshold,} if args.use_prompt else {},)
        # breakpoint()
        x, rng_key = diffusion_guidance.vanilla_denoise(
            x, params, init_t, mask, rope_index, rng_key, prompt_labels, force_drop_ids)
        # for t_i in tqdm(range(init_t)):
        #     t = init_t - t_i
        #     ### we run some eq steps first for efficient sampling
        #     for eq_step in range(N_EQ_STEPS):
        #         x, rng_key = jit_denoise_step(params, x, mask, t, rope_index, rng_key)
        #         x, rng_key = jit_noise_step(x, t, rng_key)
        #     ### x: (n_device, dbs, npt, d)
        #     x, rng_key = jit_denoise_step(params, x, mask, t, rope_index, rng_key)

        ### search steps
        cached = {
            'mask': mask, 
            'rope_index': rope_index,
            'prompt_labels': prompt_labels,
            'force_drop_ids': force_drop_ids,
            'molecules': [
                    {'smiles': init_molecules['smiles'], 
                     'graphs': init_molecules['graphs'],
                     'latents': init_molecules['latents'],},
                ], 
            'scores': [init_scores], 
            'sim': [init_sim_scores],
            'constraints': [init_constraints], 
            'unique_smiles': init_molecules['smiles'][:1], 
            'unique_scores': init_scores[:1],
            'unique_latents': init_molecules['latents'][:1],
            'predictor_params': projector_params,
            'predictor_opt_state': predictor_opt_state,
        }
        recoder.info(f'Starting search, total steps = {config.search_steps}')
        for step in range(config.search_steps - 1):
            recoder.info(f'----------------------------------------------------------------')
            recoder.info(f'Searching step {step + 1}, noise mutation steps {config.time[step]}')
            x, cached, rng_key = diffusion_es_search_step(step, x, rng_key, config, cached)
        recoder.info(f'----------------------------------------------------------------')
        recoder.info(f'Evaluating final outputs')
        
        ### decode & evaluate
        decode_molecules = decoder_f(
            x, replicate_func(cached['molecules'][-1]['smiles']),
            replicate_func(cached['molecules'][-1]['latents']),)
        scores, cached = reward_function(decode_molecules, cached)
        sim_scores = np.asarray(sim_function(
            decode_molecules['smiles'], replicate_func(cached['molecules'][0]['smiles'])), np.float32)
        constraints = constraint_function(decode_molecules, cached, config)
        cached = update_unique(cached,)

        # delete cached files
        if DELETE_CACHED_FILES:
            shutil.rmtree(os.path.join(args.save_path, f'iter_{ITER - 1}',))

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
        'constraint_weights': args.constraint_weights,
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
        'unique_smiles': cached['unique_smiles'],
        'unique_scores': cached['unique_scores'],
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
    parser.add_argument('--num_latent_tokens', type = int, default = 16)
    parser.add_argument('--dim_latent', type = int, default = 32)
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
    parser.add_argument('--t_min', type = int, required = True)
    parser.add_argument('--t_max', type = int, required = True)
    parser.add_argument('--sub_smiles', type = str, required = True)
    parser.add_argument('--on_target_scripts', type = str, nargs = '+', required = True)
    parser.add_argument('--off_target_scripts', type = str, nargs = '+', default = None)
    parser.add_argument('--affinity_predictor_params_path', type = str, default = None)
    parser.add_argument('--affinity_predictor_config_path', type = str, default = None)
    parser.add_argument('--pocket_features_path', type = str, default = None)
    parser.add_argument('--use_prompt', action = 'store_true', default = False)
    parser.add_argument('--constraint_weights', type = float, nargs = '+', default = None,)

    args = parser.parse_args()

    infer(args)