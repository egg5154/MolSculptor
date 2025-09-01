import jax.tree_util as jtu
import numpy as np
import os
from rdkit import Chem
from src.common.utils import safe_l2_normalize
from train.inference import tokens2smiles, smi2graph_features, standardize, sanitize_smiles
from train.rewards import LogP_reward, tanimoto_sim, \
    dsdp_reward, dsdp_batch_reward, QED_reward, SA_reward

def expand_batch_dim(batch_size: int, dict_array: dict):
    """
        dim expansion for diffes optimization.
    """
    def _fn(path, arr):
        if jtu.DictKey('atom_features') in path:
            ## check ndim == 2
            if arr.ndim == 2: # has batch dim
                return arr
            elif arr.ndim == 1: # no batch dim
                return arr[None].repeat(batch_size, axis = 0)
            else:
                raise
        elif jtu.DictKey('bond_features') in path:
            ## check ndim == 3
            if arr.ndim == 3:
                return arr
            elif arr.ndim == 2:
                return arr[None].repeat(batch_size, axis = 0)
            else:
                raise
        elif jtu.DictKey('scores') in path:
            ## check ndim == 1
            if arr.ndim == 2:
                return arr
            elif arr.ndim == 1:
                return arr[None].repeat(batch_size, axis = 0)
            else:
                raise
        else:
            ## check ndim == 1
            if arr.ndim == 1:
                return arr
            elif arr.ndim == 0:
                return arr[None].repeat(batch_size, axis = 0)
            else:
                raise
    
    return jtu.tree_map_with_path(_fn, dict_array)

### functions for reward calculation ###
def dual_inhibitor_reward_function(molecule_dict, cached_dict, 
                    dsdp_script_paths, save_path):
    ## for repeat molecules, we use cached scores.

    ## get unique smiles in this iter.
    unique_smiles = cached_dict['unique_smiles']
    unique_scores = cached_dict['unique_scores']
    todo_smiles = molecule_dict['smiles'] # (dbs * r,)
    todo_unique_smiles = np.unique(todo_smiles)
    todo_unique_smiles = np.setdiff1d(todo_unique_smiles, unique_smiles)
    todo_unique_scores = np.empty((0, 2), dtype = np.float32)

    # breakpoint()
    ## we use dsdp docking reward + QED reward
    if todo_unique_smiles.size > 0:
        ## run dsdp
        print('---------------PROT-1 docking---------------')
        r_dock_1 = dsdp_batch_reward(
            smiles = todo_unique_smiles,
            cached_file_path = save_path,
            dsdp_script_path = dsdp_script_paths[0],
        )
        r_dock_1 = np.asarray(r_dock_1, np.float32) * (-1.) # (N,)
        print('---------------PROT-2 docking---------------')
        r_dock_2 = dsdp_batch_reward(
            smiles = todo_unique_smiles,
            cached_file_path = save_path,
            dsdp_script_path = dsdp_script_paths[1],
            gen_lig_pdbqt = False,
        )
        r_dock_2 = np.asarray(r_dock_2, np.float32) * (-1.) # (N,)
        todo_unique_scores = np.stack([r_dock_1, r_dock_2], axis = 1) # (N, 2)
        unique_smiles = np.concatenate([unique_smiles, todo_unique_smiles])
        unique_scores = np.concatenate([unique_scores, todo_unique_scores])

    ## get score for this batch
    todo_index = [np.where(unique_smiles == s)[0][0] for s in todo_smiles]
    todo_scores = unique_scores[todo_index]
    cached_dict['update_unique_smiles'] = todo_unique_smiles
    cached_dict['update_unique_scores'] = todo_unique_scores
    return todo_scores, cached_dict

def sim_function(smiles, init_smiles,):
        sim = np.asarray(tanimoto_sim(smiles, init_smiles), dtype = np.float32,)
        return sim
    
def has_substructure(smiles, sub_smiles = None):
    assert sub_smiles is not None, 'Input arg sub_smiles is None!'
    sub_m = Chem.MolFromSmiles(sub_smiles)
    search_ms = [Chem.MolFromSmiles(s) for s in smiles]
    has_substr = [m.HasSubstructMatch(sub_m) for m in search_ms]
    return has_substr

def find_repeats(smiles, unique_smiles: np.ndarray):
    smiles_set: list = unique_smiles.tolist()
    is_repeat = []
    for s in smiles:
        if s in smiles_set:
            is_repeat.append(0)
        else:
            is_repeat.append(1)
            smiles_set.append(s)
    return np.array(is_repeat, np.int32)

### functions for encoding and decoding ###
def encoder_function(graph_features, inferencer):
        return inferencer.jit_encoding_graphs(graph_features)

def decoder_function(latent_tokens, cached_smiles, cached_latent_tokens,
                     inferencer, reverse_alphabet, beam_size=4):
        """
            For decoder g(S|z).
            NOTE: input latent tokens is scaled.
        """
        ## latent_tokens: (dbs, npt, d), cached_smiles: (dbs,)
        dbs, npt, d = latent_tokens.shape
        latent_tokens = safe_l2_normalize(
            latent_tokens / jnp.sqrt(d), axis = -1)
        output_latent_tokens = np.array(latent_tokens, jnp.float32)
        
        assert cached_smiles.shape[0] == dbs, f'{cached_smiles.shape[0]} != {dbs}'
        ## (dbs*bm, n_seq)
        output_tokens, aux = inferencer.beam_search(
            step = 0, cond = latent_tokens,)
        output_tokens = np.asarray(output_tokens, np.int32)
        ## (dbs*bm,) -> (dbs, bm)
        output_smiles = [
            tokens2smiles(t, reverse_alphabet) for t in output_tokens]
        output_smiles = np.asarray(
            output_smiles, object).reshape(dbs, beam_size)
        ## check if valid
        sanitized_output_smiles = np.empty((dbs,), object)
        count = 0
        for i_ in range(dbs):
            ## search for beam_size, from the most probable one
            ## if one is valid, then break.
            for j_ in range(beam_size - 1, -1, -1):
                smi_ = sanitize_smiles(output_smiles[i_, j_]) # standardize(output_smiles[i_, j_])
                if smi_:
                    sanitized_output_smiles[i_] = smi_
                    break
            ## if no one is valid, then use cached smiles.
            if sanitized_output_smiles[i_] is None:
                count += 1
                sanitized_output_smiles[i_] = cached_smiles[i_]
                output_latent_tokens[i_] = cached_latent_tokens[i_]
                # output_latent_tokens = output_latent_tokens.at[i_].set(cached_latent_tokens[i_])
        print('validity:', (dbs - count) / dbs)
        print('uniqueness:', len(set(sanitized_output_smiles)) / dbs)
        ## make graph features
        sanitized_output_graph_features = [
            smi2graph_features(smi,) for smi in sanitized_output_smiles]
        batched_output_graph_features = {
            top_key: {sub_key:
                np.stack([this_graph[top_key][sub_key] for this_graph in sanitized_output_graph_features]) \
                    for sub_key in sanitized_output_graph_features[0][top_key].keys()
            } for top_key in ['atom_features', 'bond_features']
        }
        return {
            'graphs': batched_output_graph_features, 
            'smiles': sanitized_output_smiles,
            'latents': output_latent_tokens,
            }

### functions for NSGA-II ###
def fast_non_dominated_sorting(scores, constraints, constraint_weights = None):
    ## scores: (N = n_pops, M = metrics)
    ## constraints: (N = n_pops, C = constraints), constraints are boolean values.

    # (N, C) -> (N,) -> (N, N)
    and_constraints_i = np.all(constraints, axis = 1).astype(np.int32)
    and_constraints_ij = and_constraints_i[:, None] + and_constraints_i[None, :]

    # cond 0: and_constraints_ij = 0
    # compare using constraints weighted sum
    # (N, C) @ (C,) -> (N,)
    n_, c_ = constraints.shape
    if constraint_weights is None: constraint_weights = np.ones(c_)
    weighted_constraints_i = constraints @ constraint_weights
    _compare_i_dom_j = (weighted_constraints_i[:, None] > weighted_constraints_i[None, :]).astype(np.int32)
    _compare_j_dom_i = (weighted_constraints_i[:, None] < weighted_constraints_i[None, :]).astype(np.int32)
    dominated_ij_0 = _compare_i_dom_j - _compare_j_dom_i

    # cond 1: and_constraints_ij = 1
    # compare using and constraints
    _compare_i_dom_j = (and_constraints_i[:, None] > and_constraints_i[None, :]).astype(np.int32)
    _compare_j_dom_i = (and_constraints_i[:, None] < and_constraints_i[None, :]).astype(np.int32)
    dominated_ij_1 = _compare_i_dom_j - _compare_j_dom_i

    # cond 2: and_constraints_ij = 2
    # compare using scores
    # (N, 1, M), (1, N, M) -> (N, N, M)
    _compare_i_dom_j = (scores[:, None] > scores[None, :]).astype(np.int32).prod(axis = 2)
    _compare_j_dom_i = (scores[:, None] < scores[None, :]).astype(np.int32).prod(axis = 2)
    dominated_ij_2 = _compare_i_dom_j - _compare_j_dom_i

    dominated_ij = dominated_ij_0 * (and_constraints_ij == 0).astype(np.int32) + \
        dominated_ij_1 * (and_constraints_ij == 1).astype(np.int32) + \
        dominated_ij_2 * (and_constraints_ij == 2).astype(np.int32)
    
    # (N, N) -> (N,)
    np_i = np.sum(dominated_ij == -1, axis = 1)
    # [(n_i,), ...]
    sp_i = [np.where(dominated_ij[_] == 1)[0] for _ in range(n_)]

    # get pareto fronts
    front_it = np.where(np_i == 0)[0]
    pareto_fronts = [front_it,]
    while front_it.size > 0:
        q_it = []
        for s in front_it:
            for q in sp_i[s]:
                np_i[q] -= 1
                if np_i[q] == 0: q_it.append(q)
        front_it = np.array(q_it, dtype = np.int32)
        pareto_fronts.append(front_it)

    return pareto_fronts

def crowding_distance_assignment(scores_it, inf = 1e6):
    ## sorting in one front set
    ## scores: (Ni, M)
    ## returns: (Ni,)

    n_i_, m_ = scores_it.shape
    distance = np.zeros(n_i_, dtype = np.float32)
    for _i in range(m_):
        front_scores_it = scores_it[:, _i] # (Ni,)
        sorted_index = np.argsort(front_scores_it)
        distance[sorted_index[0]] = inf
        distance[sorted_index[-1]] = inf
        # for i = 2 to Ni - 1
        sorted_scores = front_scores_it[sorted_index] # (Ni,)
        sorted_scores_ip1 = sorted_scores[2:]
        sorted_scores_im1 = sorted_scores[:-2]
        normed_distance_ = \
            (sorted_scores_ip1 - sorted_scores_im1) / (sorted_scores[-1] - sorted_scores[0] + 1e-12)
        distance[sorted_index[1: -1]] += normed_distance_ # (Ni - 2,)

    return distance

def NSGA_II(scores, constraints, constraint_weights = None, n_pops = 128, inf = 1e6):
    ## scores: (N = n_pops, M = metrics)
    ## constraints: (N = n_pops, C = constraints), constraints are boolean values.
    ## constraint_weights: (C,), weights for constraints.

    pareto_fronts = fast_non_dominated_sorting(scores, constraints, constraint_weights)
    next_pop_ids = np.empty(shape = (0,), dtype = np.int32)
    rank = 0
    ## strict sorting
    while next_pop_ids.size + pareto_fronts[rank].size <= n_pops:
        next_pop_ids = np.concatenate([next_pop_ids, pareto_fronts[rank]])
        rank += 1
    distance = crowding_distance_assignment(scores[pareto_fronts[rank]], inf)
    next_pop_ids = np.concatenate(
        [next_pop_ids, pareto_fronts[rank][np.argsort(distance)[::-1][:n_pops - next_pop_ids.size]]]
    )
    assert next_pop_ids.size == n_pops, \
        f'next_pop_ids.size = {next_pop_ids.size}, n_pops = {n_pops}'
    return next_pop_ids # (n_pops,)

import jax
import optax
import jax.numpy as jnp
import flax.linen as nn

from typing import Callable
from tqdm import tqdm
from ml_collections import ConfigDict
from train.scheduler import _extract_into_tensor, GaussianDiffusion

class DiffusionTFG:
    """Training free guidance utils.
    """

    def __init__(self, config: ConfigDict, dit_net: nn.Module, dit_params: dict, 
                 scheduler: GaussianDiffusion, loss_fn: Callable,):

        self.config = config
        self.dit_params = dit_params
        self.dit_net = dit_net
        self.scheduler = scheduler

        """loss_fn is a wrapped function takes only x as input.
        example: loss_fn = partial(loss_net.apply, params = your_params, aux_input = your_aux_input)
        """
        self.loss_fn = loss_fn
        self.optimizer = optax.adam(
            learning_rate=config.learning_rate,
        )

        ## tool functions
        self.jit_denoise_step = jax.jit(self.denoise_step)
        self.jit_noise_step = jax.jit(self.noise_step)
        self.jit_denoise = jax.jit(self.denoise)
        self.jit_noise = jax.jit(self.noise)
        self.jit_optimizer_init = jax.jit(self.optimizer.init)
        self.jit_grad_loss_fn = jax.jit(jax.grad(self.loss_fn, has_aux=False))
        self.jit_grad_loss_fn_xt = jax.jit(jax.grad(self.loss_fn_xt, has_aux=True))
        self._jit_scale_fn = jax.jit(self._scale_fn)
        self._jit_x_update_fn = jax.jit(self._x_update_fn)
        self._jit_noise_with_eps = jax.jit(self._noise_with_eps)
        self._jit_pred_noise_from_xstart = jax.jit(self._pred_noise_from_xstart)
        self._jit_denoise_from_eps = jax.jit(self._denoise_from_eps)
        self._jit_denoise_step_from_eps = jax.jit(self._denoise_step_from_eps)
        # self._jit_init_opt_state_and_denoise = jax.jit(self._init_opt_state_and_denoise)
        # self._jit_forward_backward_guidance_step = jax.jit(self._forward_backward_guidance_step)
        # self._jit_forward_backward_guidance_final_step = jax.jit(self._forward_backward_guidance_final_step)
    
    def denoise_step(self, params, x, mask, time, rope_index, rng_key, labels = None, force_drop_ids = None):
        time = jnp.full((x.shape[0],), time) # (dbs,)
        eps_pred = self.dit_net.apply(
            {'params': params['params']['net']}, x, mask, time, 
            labels, force_drop_ids, rope_index) # (dbs, npt, d)
        mean, variance, log_variance = self.scheduler.p_mean_variance(
            x, time, eps_pred, clamp_x0_fn = None, clip = False)
        rng_key, sub_key = jax.random.split(rng_key)
        x = mean + jnp.exp(0.5 * log_variance) * jax.random.normal(sub_key, x.shape)
        return x, rng_key
    
    def noise_step(self, x, time, rng_key):
        time = jnp.full((x.shape[0],), time) # (dbs,)
        rng_key, sub_key = jax.random.split(rng_key)
        x = self.scheduler.q_sample_step(x, time, jax.random.normal(sub_key, x.shape))
        return x, rng_key
    
    def noise(self, x, time, rng_key):
        """q(x_t | x_0)"""
        time = jnp.full((x.shape[0],), time) # (dbs,)
        rng_key, sub_key = jax.random.split(rng_key)
        x = self.scheduler.q_sample(x, time, jax.random.normal(sub_key, x.shape))
        return x, rng_key

    def denoise(self, params, x, mask, time, rope_index, labels = None, force_drop_ids = None):
        """p(x_0 | x_t)"""
        time = jnp.full((x.shape[0],), time)
        eps_pred = self.dit_net.apply(
            {'params': params['params']['net']}, x, mask, time, 
            labels, force_drop_ids, rope_index) # (dbs, npt, d)
        x_0 = self.scheduler._predict_xstart_from_eps(x, time, eps_pred)
        return x_0, eps_pred
    
    def _denoise_from_eps(self, eps, x, time):
        time = jnp.full((x.shape[0],), time)
        x_0 = self.scheduler._predict_xstart_from_eps(x, time, eps)
        return x_0
    
    def _x_update_fn(self, opt_state, x, predictor_params):
        """Update function for optimizer.
        NOTE: input x is scaled."""

        grad = jax.grad(self.loss_fn, has_aux=False)(x, predictor_params) # TODO: check here
        updates, opt_state = self.optimizer.update(grad, opt_state, x)
        x = optax.apply_updates(x, updates)
        return x, grad, opt_state
    
    def _scale_fn(self, time_step, search_step):
        """Scale function for forward guidance."""
        search_step_scale = 1 / (1 + jnp.exp(-0.5 * (search_step - 15))) # TODO: test here
        # search_step_scale = 1 # debug
        return self.scheduler.sqrt_one_minus_alphas_cumprod[time_step] * search_step_scale
    
    def _optx(self, x, opt_state, params):
        """NOTE: x is scaled(norm=dim).
        params is for the loss function, and x will be optimized."""
        x, grad_save, opt_state = self._jit_x_update_fn(opt_state, x, params) # save init grad
        for _ in range(self.config.optimization_steps - 1):
            x, _, opt_state = self._jit_x_update_fn(opt_state, x, params)
        return x, grad_save
    
    def _denoise_step_from_eps(self, x, eps, time, rng_key):
        time = jnp.full((x.shape[0],), time) # (dbs,)
        mean, var, log_var = self.scheduler.p_mean_variance(
            x, time, eps, clamp_x0_fn=None, clip=False)
        rng_key, sub_key = jax.random.split(rng_key)
        x = mean + jnp.exp(0.5 * log_var) * jax.random.normal(sub_key, x.shape)
        return x, rng_key
    
    def _pred_noise_from_xstart(self, x, time, x_0):
        time = jnp.full((x.shape[0],), time) # (dbs,)
        return (
            _extract_into_tensor(self.scheduler.sqrt_recip_alphas_cumprod, time, x.shape) * x
            - x_0
        ) / _extract_into_tensor(self.scheduler.sqrt_recipm1_alphas_cumprod, time, x.shape)
    
    def _noise_with_eps(self, x_0, time, eps):
        time = jnp.full((x_0.shape[0],), time) # (dbs,)
        """q(x_t | x_0)"""
        coeff1 = _extract_into_tensor(self.scheduler.sqrt_alphas_cumprod, time, x_0.shape)
        coeff2 = _extract_into_tensor(self.scheduler.sqrt_one_minus_alphas_cumprod, time, x_0.shape)

        return x_0 * coeff1 + eps * coeff2
    
    def loss_fn_xt(self, x_t, predictor_params, mask, time, rope_index, labels, force_drop_ids):
        x_0, eps = self.jit_denoise(self.dit_params, x_t, mask, time, rope_index, labels, force_drop_ids)
        out = self.loss_fn(x_0, predictor_params)
        return out, eps
    
    def tfg_step(self, x_t, params, time, step, mask, rope_index, rng_key, 
                 labels=None, force_drop_ids=None, renoise=True):

        try:
            # x_0, eps = self.jit_denoise(self.dit_params, x_t, mask, time, rope_index, labels, force_drop_ids)
            # g_0 = self.jit_grad_loss_fn(x_0, params)
            g_0, eps = self.jit_grad_loss_fn_xt(x_t, params, mask, time, rope_index, labels, force_drop_ids)
            # print('eps norm:', jnp.linalg.norm(eps))
            eps += self._jit_scale_fn(time, step) * g_0
            # print('g0 norm:', jnp.linalg.norm(self._jit_scale_fn(time, step) * g_0), jnp.linalg.norm(g_0))
            x_0_forward = self._jit_denoise_from_eps(eps, x_t, time)
            opt_state = self.jit_optimizer_init(x_0_forward)
            x_0_backward, _ = self._optx(x_0_forward, opt_state, params)
            # print('delta:', jnp.linalg.norm(x_0_backward - x_0_forward))
            eps_hat = self._jit_pred_noise_from_xstart(x_t, time, x_0_backward)
            x_tminus1 = self._jit_noise_with_eps(x_0_backward, time - 1, eps_hat)
            """time range: (T, T-1, ...., 1)
            """
            if renoise:
                x_t, rng_key = self.jit_noise_step(x_tminus1, time, rng_key)
                return x_t, rng_key
            else:
                return x_tminus1, rng_key
        except Exception as e:
            print(e)
            breakpoint()
    
    def tfg_denoise(
            self, x_out, params, diffusion_time_it, search_step_it, mask, 
            rope_index, rng_key, labels = None, force_drop_ids = None):

        for t_i in tqdm(range(diffusion_time_it)):
            t = diffusion_time_it - t_i
            # we run some eq steps first for efficient sampling
            for _ in range(self.config.eq_steps):
                x_out, rng_key = self.tfg_step(
                    x_out, params, t, search_step_it, mask, rope_index, rng_key, labels, force_drop_ids)
            # x: (dbs*r, npt, d)
            x_out, rng_key = self.tfg_step(
                x_out, params, t, search_step_it, mask, rope_index, rng_key, labels, force_drop_ids, renoise=False)
        return x_out, rng_key
    
    def vanilla_denoise(
            self, x, params, diffusion_time_it, mask, rope_index, 
            rng_key, labels = None, force_drop_ids = None):
        
        for t_i in tqdm(range(diffusion_time_it)):
            t = diffusion_time_it - t_i
            ### we run some eq steps first for efficient sampling
            for eq_step in range(self.config.eq_steps):
                x, rng_key = self.jit_denoise_step(params, x, mask, t, rope_index, rng_key, labels, force_drop_ids)
                x, rng_key = self.jit_noise_step(x, t, rng_key)
            ### x: (n_device, dbs, npt, d)
            x, rng_key = self.jit_denoise_step(params, x, mask, t, rope_index, rng_key, labels, force_drop_ids)
        return x, rng_key
    
    def generate_input_labels(self, npops: int, nreps: int, constraints: dict):
        qed_min, qed_max = 0.4, 0.9
        sas_min, sas_max = 2.0, 4.5
        logp_min, logp_max = 0.0, 6.0
        mw_min, mw_max = 200., 600.
        input_labels = {}
        if constraints.get('qed_min', None) is not None:
            qed_min = max(constraints.pop('qed_min'), qed_min)
            input_labels['qed'] = np.tile(np.linspace(qed_min, qed_max, npops), nreps)
        else:
            input_labels['qed'] = np.full(npops*nreps, -1e5)
        if constraints.get('sas_max', None) is not None:
            sas_max = min(constraints.pop('sas_max'), sas_max)
            input_labels['sa'] = np.tile(np.linspace(sas_min, sas_max, npops), nreps)
        else:
            input_labels['sa'] = np.full(npops*nreps, -1e5)
        if (constraints.get('logp_min', None) is not None) or (constraints.get('logp_max', None) is not None):
            logp_min = max(constraints.pop('logp_min', logp_min), logp_min)
            logp_max = min(constraints.pop('logp_max', logp_max), logp_max)
            input_labels['logp'] = np.tile(np.linspace(logp_min, logp_max, npops), nreps)
        else:
            input_labels['logp'] = np.full(npops*nreps, -1e5)
        if (constraints.get('mw_min', None) is not None) or (constraints.get('mw_max', None) is not None):
            mw_min = max(constraints.pop('mw_min', mw_min), mw_min)
            mw_max = min(constraints.pop('mw_max', mw_max), mw_max)
            input_labels['mw'] = np.tile(np.linspace(mw_min, mw_max, npops), nreps)
        else:
            input_labels['mw'] = np.full(npops*nreps, -1e5)
        if len(constraints) > 0:
            raise Warning(f'Those constraints are not supported or needed: {list(constraints.keys())}')
        force_drop_ids = jnp.array([0,] * (npops*nreps), jnp.int32) ## no drop
        return input_labels, force_drop_ids

    
    ## deprecated
    # def _init_opt_state_and_denoise(self, x_t, mask, time, rope_index):
    #     x_0, eps = self.denoise(self.dit_params, x_t, mask, time, rope_index)
    #     init_opt_state = self.optimizer.init(x_0)
    #     return init_opt_state, x_0, eps
    
    # def _forward_backward_guidance_step(self, x_t, dx_0, g_0, eps, time, step, rng_key):

    #     # forward guidance
    #     eps += self._scale_fn(time, step) * g_0
    #     # backward guidance
    #     eps += (-1) * dx_0 * _extract_into_tensor(1 / self.scheduler.sqrt_recipm1_alphas_cumprod, time, eps.shape)
    #     x_t, rng_key = self._denoise_step_from_eps(x_t, eps, time, rng_key)
    #     x_t, rng_key = self.noise_step(x_t, time, rng_key)
    #     return x_t, rng_key
    
    # def tfg_step(self, x_t, params, time, step, mask, rope_index, rng_key):
    
    #     # init_opt_state = self.jit_optimizer_init(x_t)
    #     # x_0, eps = self.jit_denoise(self.dit_params, x_t, mask, time, rope_index)
    #     init_opt_state, x_0, eps = self._jit_init_opt_state_and_denoise(x_t, mask, time, rope_index)
    #     x_0_opt, grad_0 = self._optx(x_0, init_opt_state, params) # no jit in for loop
    #     breakpoint()
    #     dx_0 = x_0_opt - x_0 # (dbs, npt, d)
    #     x_t, rng_key = self._jit_forward_backward_guidance_step(x_t, dx_0, grad_0, eps, time, step, rng_key)

    #     return x_t, rng_key
    
    # def _forward_backward_guidance_final_step(self, x_t, dx_0, g_0, eps, time, step, rng_key):

    #     # forward guidance
    #     eps += self._scale_fn(time, step) * g_0
    #     # backward guidance
    #     eps += (-1) * dx_0 * _extract_into_tensor(1 / self.scheduler.sqrt_recipm1_alphas_cumprod, time, eps.shape)
    #     x_t_minus_1, rng_key = self._denoise_step_from_eps(x_t, eps, time, rng_key)
    #     return x_t_minus_1, rng_key

    # def tfg_final_step(self, x_t, params, mask, time, step, rope_index, rng_key):
    #     init_opt_state, x_0, eps = self._jit_init_opt_state_and_denoise(x_t, mask, time, rope_index)
    #     x_0_opt, grad_0 = self._optx(x_0, init_opt_state, params) # no jit in for loop
    #     dx_0 = x_0_opt - x_0 # (dbs, npt, d)
    #     x_t_minus_1, rng_key = self._jit_forward_backward_guidance_final_step(
    #         x_t, dx_0, grad_0, eps, time, step, rng_key)
        
    #     return x_t_minus_1, rng_key

    # def tfg_denoise(
    #         self, x_out, params, diffusion_time_it, search_step_it, mask, rope_index, rng_key):

    #     for t_i in tqdm(range(diffusion_time_it)):
    #         t = diffusion_time_it - t_i
    #         # we run some eq steps first for efficient sampling
    #         for _ in range(self.config.eq_steps):
    #             x_out, rng_key = self.tfg_step(
    #                 x_out, params, t, search_step_it, mask, rope_index, rng_key)
    #         # x: (dbs*r, npt, d)
    #         x_out, rng_key = self.tfg_final_step(
    #             x_out, params, mask, t, search_step_it, rope_index, rng_key)
    #     return x_out, rng_key

import copy
import math
import datetime
import pickle as pkl
from logging import Logger

Optimizer = optax.GradientTransformation
class DockingSurrogateModel:

    def __init__(self, config: ConfigDict, projector_net: nn.Module, optimizer: Optimizer, recoder: Logger,
                 encoder_net: nn.Module = None, encoder_params: dict = None,):
        
        self.config = config
        self.encoder = encoder_net
        self.net = projector_net
        self.eval_net: nn.Module = copy.deepcopy(projector_net)
        self.eval_net.global_config.dropout_flag = False
        self.optimizer = optimizer
        self.recoder = recoder
        self.split_rate = config.train.train_eval_split_rate

        if self.encoder:
            assert encoder_params is not None
            # assert self.net.config.input_scaled == False, 'Input is not scaled while encoder is used!'
            self.encoder.global_config.dropout_flag = False
            self.encoder_params = jtu.tree_map(lambda x: jnp.asarray(x, jnp.float32), encoder_params)

        self._forward_and_backward = jax.value_and_grad(self._forward, has_aux=True)
        self._jit_forward = jax.jit(self._forward)
        self._jit_train_one_step = jax.jit(self._train_one_step)
        self._jit_eval_net_apply = jax.jit(self.eval_net.apply)
        self._jit_eval_fn = jax.jit(self._eval_fn)

        self.train_data = {
            'inputs': jnp.empty((0, config.npt, config.dim), jnp.float32),
            'labels': jnp.empty((0, config.score_weights.shape[0]), jnp.float32),
        }
        self.eval_data = {
            'inputs': jnp.empty((0, config.npt, config.dim), jnp.float32),
            'labels': jnp.empty((0, config.score_weights.shape[0]), jnp.float32),
        }
    
    def add_data(self, x: jnp.ndarray, y: jnp.ndarray):
        """add train / eval data.
            x: latent_tokens (NOTE: scaled), shape of (n_data, npt, d)
            y: labels, shape of (n_data, out)
        """
        assert x.shape[0] == y.shape[0], 'x and y must have same data size.'
        assert jnp.allclose(jnp.sum(x ** 2, axis = -1), x.shape[-1], rtol=0.05), 'x must be scaled.'
        # assert jnp.all(y >= 0), 'y must be non-negative.'
        _data = {'inputs': x, 'labels': y}
        """
            We need to filt trash here.
        """
        _filt_idx = np.prod(y > 0, -1).astype(np.bool_)
        _data = jtu.tree_map(lambda x: x[_filt_idx], _data)
        _train_data, _eval_data = self.split_data(_data,)
        # self.train_data = jtu.tree_map(
        #     lambda x, y: jnp.concatenate([x, y], axis=0), self.train_data, _train_data)
        # self.eval_data = jtu.tree_map(
        #     lambda x, y: jnp.concatenate([x, y], axis=0), self.eval_data, _eval_data)
        self.train_data = _train_data
        self.eval_data = _eval_data
        self.recoder.info('Add train / eval data...')
        _train_size = self.train_data['labels'].shape[0]
        _eval_size = self.eval_data['labels'].shape[0]
        self.recoder.info(f'\tCurrent train data size: {_train_size}')
        self.recoder.info(f'\tCurrent eval data size: {_eval_size}')
    
    def split_data(self, data: dict,):
        """split data into train and eval sets.
            data: dict, keys are 'inputs' and 'labels'.
        """
        n_data = data['inputs'].shape[0]
        n_train = int(n_data * self.split_rate)
        indices = np.arange(n_data)
        np.random.shuffle(indices)
        train_indices = indices[:n_train]
        eval_indices = indices[n_train:]
        
        train_data = jtu.tree_map(lambda x: x[train_indices], data)
        eval_data = jtu.tree_map(lambda x: x[eval_indices], data)
        
        return train_data, eval_data
    
    def init_net(self, rng_key, npt, dim):
        init_data = jax.random.normal(rng_key, (1, npt, dim))
        if self.encoder is not None:
            init_data = self.encoder.apply(self.encoder_params, init_data)
        net_params = self.net.init({'params': rng_key, 'dropout': rng_key}, init_data,)
        return net_params
    
    def init_optimizer(self, net_params):
        return self.optimizer.init(net_params)
    
    def _forward(self, params, batch_data, rng_key):
        x, y = batch_data['inputs'], batch_data['labels']
        drop_key, rng_key = jax.random.split(rng_key)
        if self.encoder is not None:
            x = self.encoder.apply(self.encoder_params, x) # encoder params is fixed
        y_hat = self.net.apply(params, x, rngs={'dropout': drop_key}) # (bs, out)
        # y_hat = y_hat * self.config.scale + self.config.shift
        # mse loss
        assert y.shape == y_hat.shape, f'y shape {y.shape} != y_hat shape {y_hat.shape}'
        loss = jnp.mean((jnp.square(y - y_hat)))
        return loss, (y_hat, rng_key)
    
    def _train_one_step(self, batch_data, params, opt_state, rng_key):
        loss, grad = self._forward_and_backward(params, batch_data, rng_key)
        loss, (_, rng_key) = loss
        params_update, opt_state = self.optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, params_update)
        return loss, params, opt_state, rng_key

    def train(self, rng_key, net_params, opt_state, save_path,):
        
        max_train_step = 50000 # TODO: add this to config
        train_data, eval_data = self.train_data, self.eval_data
        bs = self.config.train.batch_size
        n_data = train_data['inputs'].shape[0]
        n_steps_per_epoch = math.ceil(n_data / bs)
        n_steps = math.ceil(n_data / bs) * self.config.train.num_epochs
        n_steps = min(n_steps, max_train_step)

        best_eval_loss = self.eval(net_params, eval_data)
        best_params, best_opt_state = net_params, opt_state
        os.makedirs(os.path.join(save_path, 'params'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'opt_states'), exist_ok=True)

        self.recoder.info(f'Start training surrogate model, total steps = {n_steps}')
        start_time = datetime.datetime.now()
        for step in range(n_steps):
            start_idx = step * bs
            stop_idx = start_idx + bs
            batch_data = jtu.tree_map(lambda x: x[np.arange(start_idx, stop_idx) % n_data], train_data)
            loss, net_params, opt_state, rng_key = self._jit_train_one_step(
                batch_data, net_params, opt_state, rng_key)
        
            if (step + 1) % n_steps_per_epoch == 0:
                eval_loss = self.eval(net_params, eval_data)
                self.recoder.info(
                    f"Step {step + 1:<5d} | current loss: {loss:.4f}, eval loss: {eval_loss:.4f}.")
                if eval_loss < best_eval_loss:
                    best_eval_loss, best_params, best_opt_state = eval_loss, net_params, opt_state
            
            if (step + 1) % n_steps_per_epoch == 0:
                save_path_ = os.path.join(save_path, f'params/params_step{step + 1}.pkl')
                with open(save_path_, 'wb') as f:
                    pkl.dump(jtu.tree_map(lambda arr: np.asarray(arr, np.float32), net_params), f)
                save_path_ = os.path.join(save_path, f'opt_states/opt_state_step{step + 1}.pkl')
                with open(save_path_, 'wb') as f:
                    pkl.dump(jtu.tree_map(lambda arr: np.asarray(arr, np.float32), opt_state), f)
            
            if (step + 1) % math.ceil(n_data / bs) == 0:
                permutation_index = np.random.permutation(n_data)
                train_data = jtu.tree_map(lambda arr: arr[permutation_index], train_data)
        end_time = datetime.datetime.now()
        self.recoder.info(f"Training time: {end_time - start_time}, best eval loss: {best_eval_loss:.4f}.")
        return rng_key, best_params, best_opt_state

    def _eval_fn(self, params, batch_inputs):
        if self.encoder is not None:
            batch_inputs = self.encoder.apply(self.encoder_params, batch_inputs)
        pred_ = self.eval_net.apply(params, batch_inputs) # (bs, out)
        return pred_

    def eval(self, params, eval_data):

        bs = self.config.eval.batch_size
        n_data = eval_data['inputs'].shape[0]
        n_steps = math.ceil(n_data / bs)
        pred_labels = []

        for step in range(n_steps):
            start_idx = step * bs
            stop_idx = start_idx + bs
            batch_inputs = eval_data['inputs'][np.arange(start_idx, stop_idx) % n_data]
            pred_ = self._jit_eval_fn(params, batch_inputs,) # (bs, out)
            # pred_ = pred_ * self.config.scale + self.config.shift
            pred_labels.append(pred_)
        pred_labels = jnp.concatenate(pred_labels, axis=0)[:n_data] # (n_data, out)
        eval_loss = jnp.mean((pred_labels - eval_data['labels']) ** 2)
        return eval_loss
    
    def tfg_loss_fn(self, inputs, params): # TODO: modify to support both pretrained and no pretrained.
        """loss function for training free guidance.
        inputs: (bs, npt, d), weights: (out,)"""
        
        weights = self.config.score_weights
        pred_scores = self._eval_fn(params, inputs) # (bs, out)
        """
            pred labels are positive, and we use score weights to distinguish between
            on-targets and off-targets.
        """
        assert pred_scores.shape[-1] == weights.shape[0], \
            f'pred_scores.shape[-1] = {pred_scores.shape[-1]}, weights.shape = {weights.shape[0]}'
        weighted_pred_scores = jnp.sum(pred_scores * weights[None, :], axis = -1) # (bs,)
        return jnp.sum(weighted_pred_scores)

import logging
def set_recoder(logger_path: str) -> Logger:

    #### set recoder
    recoder = logging.getLogger("inferencing dit")
    recoder.setLevel(level = logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level = logging.DEBUG)
    recoder.addHandler(stream_handler)
    file_handler = logging.FileHandler(logger_path)
    file_handler.setLevel(level = logging.DEBUG)
    recoder.addHandler(file_handler)
    recoder.propagate = False

    return recoder

from typing import Tuple
def load_infer_config(config_path: str) -> Tuple[ConfigDict, ConfigDict, ConfigDict, ConfigDict]:
    with open(config_path, 'rb') as f:
        config_dicts = pkl.load(f)
    global_config = ConfigDict(config_dicts['global_config'])
    net_config = ConfigDict(config_dicts['net_config'])
    train_config = ConfigDict(config_dicts['train_config'])
    data_config = ConfigDict(config_dicts['data_config'])
    global_config.dropout_flag = False
    return net_config, train_config, data_config, global_config

def load_infer_params(dit_params_path: str, vae_params_path: str) -> Tuple[dict, dict, dict]:
    with open(vae_params_path, 'rb') as f:
        vae_params = pkl.load(f)
        vae_params = jtu.tree_map(lambda x: jnp.asarray(x), vae_params)
    encoder_params = {
        'Encoder_0': vae_params['params']['generator']['Encoder_0'],
        'Dense_0': vae_params['params']['generator']['Dense_0'],
    }
    decoder_params = {
        'Decoder_0': vae_params['params']['generator']['Decoder_0'],
    }
    with open(dit_params_path, 'rb') as f:
        params = pkl.load(f)
        params = jtu.tree_map(jnp.asarray, params)
    return params, encoder_params, decoder_params

from argparse import ArgumentParser
def set_infer_config(args: ArgumentParser, data_config: ConfigDict,) -> ConfigDict:
    default_infer_config = {
        'sampling_method': args.sampling_method,
        'device_batch_size': args.device_batch_size,
        'n_seq_length': data_config['n_pad_token'],
        'beam_size': args.beam_size,
        'bos_token': data_config['bos_token_id'],
        'eos_token': data_config['eos_token_id'],
        'n_local_device': jax.local_device_count(),
        'num_prefix_tokens': data_config['n_query_tokens'],
        'step_limit': 160,
    }
    return ConfigDict(default_infer_config)

# from src.common.layers.mlp import MLP
# class Predictor(nn.Module):

#     config: ConfigDict

#     @nn.compact
#     def __call__(self, latents,):
#         # latents: (dbs, npt, d) NOTE: scaled().
#         arr_dtype = jnp.bfloat16 if self.config.bf16_flag else jnp.float32
#         dbs, npt, d = latents.shape
#         latents = safe_l2_normalize(latents / jnp.sqrt(d), axis=-1)
#         mlp_net = MLP(
#             output_sizes=self.config.output_sizes,
#             activation=self.config.activation,
#             dtype=arr_dtype,
#             dropout_rate=self.config.dropout_rate,
#             dropout_flag=self.config.dropout_flag,
#         )
#         mlp_net_lists = [mlp_net,] * self.config.n_scores
#         latents = jnp.mean(latents, axis=-2) # (dbs, d)
#         outputs = [mlp_net_lists[i](latents) for i in range(self.config.n_scores)]
#         outputs = jnp.concatenate(outputs, axis=-1) # (dbs, n_scores)
#         return outputs

def set_tfg_config(score_weights, eq_steps, npt, dim) -> Tuple[ConfigDict, ConfigDict]:

    # predictor_config = {
    #     'bf16_flag': True,
    #     'output_sizes': [64, 32, 1],
    #     'activation': 'leaky_relu',
    #     'dropout_rate': 0.2,
    #     'dropout_flag': True,
    #     'n_scores': score_weights.shape[0],
    # }

    surrogate_model_config = {
        # 'predictor': predictor_config,
        # 'scale': scale,
        # 'shift': shift,
        'train': {
            'train_eval_split_rate': 0.9,
            'batch_size': 64,
            'num_epochs': 150,
            'learning_rate': 0.001,
            'weight_decay': 1e-3,
        },
        'eval': {
            'batch_size': 32,
        },
        'score_weights': score_weights,
        'npt': npt,
        'dim': dim,
        # 'save_path': save_path,
    }

    tfg_config = {
        'learning_rate': 0.001,
        'optimization_steps': 50,
        'eq_steps': eq_steps,
    }

    return ConfigDict(surrogate_model_config), ConfigDict(tfg_config)

