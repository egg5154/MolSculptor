import jax.tree_util as jtu
import numpy as np
import os
from rdkit import Chem
from src.common.utils import safe_l2_normalize
from train.inference import tokens2smiles, smi2graph_features, standardize
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

def decoder_function(latent_tokens, cached_smiles, inferencer, reverse_alphabet, beam_size=4):
        """
            For decoder g(S|z).
        """
        ## latent_tokens: (dbs, npt, d), cached_smiles: (dbs,)
        dbs, npt, d = latent_tokens.shape
        assert cached_smiles.shape[0] == dbs, f'{cached_smiles.shape[0]} != {dbs}'
        latent_tokens = safe_l2_normalize(
            latent_tokens / np.sqrt(d), axis = -1)
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
                smi_ = standardize(output_smiles[i_, j_])
                if smi_:
                    sanitized_output_smiles[i_] = smi_
                    break
            ## if no one is valid, then use cached smiles.
            if not sanitized_output_smiles[i_]:
                count += 1
                sanitized_output_smiles[i_] = cached_smiles[i_]
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
            'smiles': sanitized_output_smiles
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