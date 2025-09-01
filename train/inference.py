import os
import sys
import jax
import copy
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

sys.path.append(os.path.dirname(sys.path[0]))

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.rdBase import BlockLogs
from jax import Array
from jax.tree_util import tree_map
from flax.jax_utils import replicate, unreplicate
from flax.linen.initializers import truncated_normal
from typing import Union, Optional, Tuple
from ml_collections.config_dict import ConfigDict
from net.encoder import Encoder
from net.decoder import Decoder
from src.common.utils import safe_l2_normalize

def softmax_cross_entropy(logits, label, seq_mask):
    
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(log_probs * label, axis=-1)
    
    return jnp.sum(loss * seq_mask, axis=-1)

def greedy_search(logits, previous_tokens, step_it):

    ### logits: (batch, n_seq - n_q, n_token)
    pred_token = jnp.argmax(logits, -1) # (batch_size, n_seq - n_q)
    output_tokens = previous_tokens.at[:, step_it + 1].set(pred_token[:, step_it])
    return output_tokens

def beam_search(logits, previous_log_probs, previous_tokens, step_it, 
                search_mask, beam_size = 4,) -> Tuple[Array, Array]:

    def _take(sequence_beam, sequence_id):
        return sequence_beam[sequence_id]

    ### logits: (batch * beam_size, n_ = n_seq - num_prefix_tokens, n_token)
    ### previous_log_probs: (batch, beam_size)
    ### previous_tokens: (batch*beam_size, n_seq)

    bsbm, n, n_token = logits.shape
    previous_tokens = previous_tokens[:, :n]
    logits = logits.reshape(-1, beam_size, n, n_token) # (bs, bm, n, n_token)
    logits = logits[:, :, step_it, :] # (bs, bm, n_token)
    log_prob_last_token = jax.nn.log_softmax(logits, -1) # (bs, bm, n_token)
    ### we get top k log probs and corresponding token ids
    # (batch, beam_size, n_token) -> (batch, beam_size, beam_size)
    # log_prob_last_token = log_prob[:, :, step_it, :] 
    log_prob_last_token_topk = jnp.sort(log_prob_last_token, -1)[:, :, -beam_size:]
    last_token_topk = jnp.argsort(log_prob_last_token, -1)[:, :, -beam_size:] # (bs, bm, bm)
    # (..., beam_size, 1) + (..., beam_size, beam_size)
    ### Log(p1*p2*...pt-1) + Log(pt)
    log_prob_accum = previous_log_probs[..., None] + log_prob_last_token_topk
    log_prob_accum = log_prob_accum.reshape(-1, beam_size * beam_size) # (bs, bm * bm)
    log_prob_accum += (1. - search_mask[None, :]) * (-1e5)
    tokens_accum = previous_tokens[:, None, :] # (bs * bm, 1, n)
    ## (batch_size*beam_size, beam_size, n)
    tokens_accum = jnp.tile(tokens_accum, (1, beam_size, 1)).reshape(-1, beam_size, beam_size, n) # (bs, bm, bm, n)
    tokens_accum = tokens_accum.at[:, :, :, step_it + 1].set(last_token_topk)
    tokens_accum = tokens_accum.reshape(-1, beam_size * beam_size, n) # (bs, bm * bm, n)
    top_k_sequence_id = jnp.argsort(log_prob_accum, -1)[:, -beam_size:] # (bs, bm)
    # (batch_size, beam_size*beam_size, n) -> (batch_size, beam_size, n)
    output_tokens = jax.vmap(_take, in_axes=(0, 0))(tokens_accum, top_k_sequence_id)
    # (batch_size, beam_size*beam_size) -> (batch_size, beam_size)
    output_log_prob_accum = jax.vmap(_take, in_axes=(0, 0))(log_prob_accum, top_k_sequence_id)
    # output shape: (batch_size*beam_size, n), (batch_size, beam_size)
    return output_tokens.reshape(-1, n), output_log_prob_accum

### TODO: check it
def top_p_search(logits, p, rng_key, previous_tokens, step_it):
    ### logits: (bs = batch_size, n = n_seq - n_prefix, n_token)
    ### previous_tokens: (bs, n_seq)
    bs, n, n_token = logits.shape
    logits = logits[:, step_it, :] # (bs, n_token)
    sorted_indices = jnp.argsort(logits, -1) # (bs, n_token)
    sorted_probs = jax.nn.softmax(logits, -1) # (bs, n_token)
    cum_probs = jnp.cumsum(sorted_probs, -1) # (bs, n_token)
    mask = cum_probs > p
    eps_probs = jnp.concatenate(
        [jnp.zeros((bs, n_token - 1)), jnp.ones((bs, 1)) * 1e-6], axis = -1
    )
    mask += eps_probs # (0, 0, ..., 1, 1, ..., 1 + eps)
    mask = mask / mask.sum(-1, keepdims = True) # (bs, n_token)
    selected_token_ids = jax.random.choice(
        rng_key, sorted_indices, p = mask, axis = -1,
    ) # (bs,)
    previous_tokens = previous_tokens.at[:, step_it + 1].set(selected_token_ids)
    return previous_tokens[:, :n]

### TODO: check it
def top_k_search(logits, k, rng_key, previous_tokens, step_it):
    ### logits: (bs = batch_size, n = n_seq - n_prefix, n_token)
    ### previous_tokens: (bs, n_seq)
    bs, n, n_token = logits.shape
    logits = logits[:, step_it, :] # (bs, n_token)
    top_k_indices = jnp.argsort(logits, -1)[:, -k:] # (bs, k)
    top_k_logits = logits[top_k_indices]
    top_k_log_probs = jax.nn.log_softmax(top_k_logits, -1) # (bs, k)
    top_k_probs = jnp.exp(top_k_log_probs)
    selected_token_ids = jax.random.choice(
        rng_key, top_k_indices, p = top_k_probs, axis = -1,
    ) # (bs,)
    previous_tokens = previous_tokens.at[:, step_it + 1].set(selected_token_ids)
    return previous_tokens[:, :n]

def tokens2smiles(tokens, reverse_alphabet):
    ### tokens: (n_seq,)
    smiles = []
    for t in tokens[1:]: ## skip BOS
        if reverse_alphabet[t] == '[EOS]':
            break
        smiles.append(reverse_alphabet[t])
    return ''.join(smiles)

#### encoder for inferencing
class InferEncoder(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, atom_features, bond_features,):

        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        graph_feat = Encoder(
            self.config.encoder, self.global_config,
        )(atom_features, bond_features)
        graph_feat = nn.Dense(
            self.config.latent_dim, kernel_init = truncated_normal(0.01),
            dtype = arr_dtype, use_bias = False, param_dtype = jnp.float32,
        )(graph_feat)

        ### remove sink token
        # breakpoint() ## check here
        num_sink_tokens = self.config.num_sink_tokens
        graph_feat = graph_feat[:, :-num_sink_tokens]
        graph_feat = safe_l2_normalize(graph_feat, axis = -1) ## do l2 norm for every tokens
        return graph_feat

class InferDecoder(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, sequence_features, graph_feat):
        #### shape: sequence_features: (B, N), graph_feat: (B, NQ, F)

        #### (B, N)
        seq_tokens = sequence_features['tokens']
        seq_mask = sequence_features['mask']
        batch_size, num_tokens = seq_tokens.shape
        seq_rope_index = jnp.arange(num_tokens, dtype=jnp.int32)[None, :]
        seq_rope_index = jnp.broadcast_to(seq_rope_index, seq_tokens.shape)

        #### predict logits: (B, N, L)
        # print(jax.tree_util.tree_map(jnp.shape, (seq_tokens, seq_mask, seq_rope_index, graph_feat)))
        seq_logits = Decoder(
            self.config.decoder, self.global_config
            )(seq_tokens, seq_mask, seq_rope_index, graph_feat)
        
        return seq_logits

class Inferencer:

    def __init__(self, 
                 encoding_net: nn.Module, 
                 decoding_net: nn.Module, 
                 encoding_params: dict,
                 decoding_params: dict,
                 config: ConfigDict,):

        self.encoding_net = encoding_net ## net instance after init
        self.decoding_net = decoding_net
        self.encoding_params = encoding_params
        self.decoding_params = decoding_params
        self.config = config
        self.sampling_method = config.sampling_method
        assert self.sampling_method in ['greedy', 'beam', 'top_p', 'top_k', 'nucleus']

        self.encoding_function = jax.jit(encoding_net.apply)
        self.decoding_function = jax.jit(decoding_net.apply)

        self.jit_sample_per_step = jax.jit(self._sample_per_step)
        self.jit_encoding_graphs = jax.jit(self.encoding_graphs)

        self.pmap_sample_per_step = jax.pmap(self._sample_per_step, axis_name='i')
        self.pmap_encoding_graphs = jax.pmap(self.encoding_graphs, axis_name='i')

        #### setting constants
        self.device_batch_size = config.device_batch_size
        self.n_local_device = jax.local_device_count()
        self.n_seq_length = config.n_seq_length
    
    def _sample_per_step(self, input_tokens_it, step_it, mask_it, cached_arr):

        ## cond_feat: (batch_size, num_prefix_tokens)
        ## input_tokens_it: (batch_size, n_seq)

        #### make mask & rope index & cond feature
        batch_size, n_seq = input_tokens_it.shape
        cond = cached_arr['cond']
        _, num_prefix_tokens, _ = cond.shape
        #### infer
        output_logits = self.decoding_function(
            {'params': self.decoding_params}, 
            {'tokens': input_tokens_it, 'mask': mask_it}, cond
        ) # (batch_size, n_seq, vocab_size)
        output_logits = output_logits[:, num_prefix_tokens:, :] # (batch_size, n, n_tokens)

        #### sampling
        if self.sampling_method == 'greedy':
            output_tokens = greedy_search(output_logits, input_tokens_it, step_it)

        elif self.sampling_method == 'beam':
            beam_size = self.config.beam_size
            output_tokens, output_log_prob_sum = beam_search(
                output_logits, cached_arr['log_prob_sum'], input_tokens_it, step_it, 
                cached_arr['search_mask'], beam_size,
            )
            cached_arr['log_prob_sum'] = output_log_prob_sum
        elif self.sampling_method == 'top_k':
            output_tokens = top_k_search(
                output_logits, self.config.top_k, cached_arr['rng_key'], input_tokens_it, step_it,
            )
        elif self.sampling_method == 'top_p':
            output_tokens = top_p_search(
                output_logits, self.config.top_p, cached_arr['rng_key'], input_tokens_it, step_it,
            )
        else:
            raise NotImplementedError(f"Sampling method {self.sampling_method} is not implemented")

        output_tokens = jnp.pad(output_tokens, ((0, 0), (0, n_seq - output_tokens.shape[-1])), mode='constant', constant_values=1) ## pad eos
        return output_tokens, cached_arr

    def encoding_graphs(self, graph_features):

        atom_features = graph_features['atom_features']
        bond_features = graph_features['bond_features']
        ## (batch_size, num_query_tokens, dim_feat)
        return self.encoding_function({'params': self.encoding_params}, atom_features, bond_features)

    def beam_search(self, step, cond, input_tokens = None):
        ### cond: (batch_size, num_prefix_tokens, dim_feat)

        n_seq = self.n_seq_length
        beam_size = self.config.beam_size
        batch_size = cond.shape[0]
        cached_arr_ = {}
        if input_tokens is None:
            assert step == 0
            ### create bos + eos input: (bs*bm, n_seq)
            input_tokens = np.full((batch_size*beam_size, 1), self.config.bos_token, dtype=np.int32)
            input_tokens = np.concatenate(
                [
                    input_tokens, 
                    np.full((batch_size*beam_size, n_seq - 1), self.config.eos_token, dtype=np.int32)
                    ],
                axis = -1,
            )
        ## make cond feat
        cached_arr_['log_prob_sum'] = np.zeros((batch_size, beam_size))
        _shape = (-1,) + cond.shape[1:]
        cond = np.tile(cond[:, None], (1, beam_size, 1, 1)).reshape(_shape) ## (bs*bm, n_prefix, dim_feat)
        cached_arr_['cond'] = cond

        ## sampling
        step_limit = self.config.step_limit
        for step_it in range(step, step_limit):
            
            if step_it == 0: ## first step for beam search
                cached_arr_['search_mask'] = np.pad(
                    np.ones((beam_size,), np.int32), (0, beam_size * beam_size - beam_size), mode='constant', constant_values=0,
                )

            mask = np.ones((batch_size*beam_size, step_it + 1))
            mask = np.pad(mask, ((0, 0), (0, n_seq - step_it - 1)), mode='constant', constant_values=0)

            ### jit per step
            input_tokens, cached_arr_ = self.jit_sample_per_step(input_tokens, step_it, mask, cached_arr_) 

            if step_it == 0:
                cached_arr_['search_mask'] = np.ones((beam_size * beam_size,), np.int32)
            
            ### eos check
            this_tokens = input_tokens[:, step_it + 1] # (bs*bm,)
            if (this_tokens == self.config.eos_token).all():
                break

        return input_tokens, cached_arr_
    
    def beam_search_multi_device(self, step, cond, input_tokens = None):
        #### cond: (n_device, batch_size, n_prefix, dim_feat)

        ###### constants
        dbs = self.device_batch_size
        nseq = self.n_seq_length
        nd = self.n_local_device
        beam_size = self.config.beam_size

        ###### init
        cached_arr_ = {}
        cond = np.expand_dims(cond, 2).repeat(
            beam_size, 2).reshape((nd, -1,) + cond.shape[2:]) # (n_device, bs*bm, n_prefix, d)
        cached_arr_['cond'] = cond

        ## create input tokens
        if input_tokens is None:
            assert step == 0
            ### create bos + eos input
            input_tokens = np.full((dbs * beam_size, 1), self.config.bos_token, dtype=np.int32)
            input_tokens = np.concatenate(
                [
                    input_tokens, 
                    np.full((dbs * beam_size, nseq - 1), self.config.eos_token, dtype=np.int32)
                    ],
                axis = -1,
            ) # (bs*bm, n_seq), (bos, eos, eos, ..., eos)
        input_tokens = replicate(input_tokens) # (n_device, bs*bm, n_seq)
        ## make rope index & cond
        log_prob_sum_ = np.zeros((dbs, beam_size))
        cached_arr_['log_prob_sum'] = replicate(log_prob_sum_)
        
        ## sampling
        step_limit = self.config.step_limit
        for step_it in range(step, step_limit):

            if step_it == 0: ## first step for beam search
                search_mask_ = np.pad(
                    np.ones((beam_size,), np.int32), (0, beam_size * beam_size - beam_size), mode='constant', constant_values=0,
                )
                cached_arr_['search_mask'] = replicate(search_mask_) # (n_device, beam_size*beam_size,)
            
            # (n_device, bs*bm, n_seq)
            mask = np.ones((dbs * beam_size, step_it + 1))
            mask = np.pad(mask, ((0, 0), (0, nseq - step_it - 1)), mode='constant', constant_values=0)
            mask = replicate(mask)

            ### jit per step
            input_step_it = replicate(step_it)
            input_tokens, cached_arr_ = self.pmap_sample_per_step(input_tokens, input_step_it, mask, cached_arr_)

            if step_it == 0:
                search_mask_ = np.ones((beam_size * beam_size,), np.int32)
                cached_arr_['search_mask'] = replicate(search_mask_)
            
            ### eos check: input_tokens shape (n_device, bs*bm, n_seq)
            this_tokens = input_tokens[:, step_it + 1] # (n_device, bs*bm,)
            stop_flag = (this_tokens == self.config.eos_token).all()
            if stop_flag:
                break

        return input_tokens, cached_arr_

##############################################
###### UTILS FOR MAKE GRAPH FEATURES #########
##############################################
# def standardize(smiles, dropout_threshold = 0.7):
#     """
#         Standardize smiles string, return None if failed.
#         Standardization includes:
#             1. clean up;
#             2. remove sub fragments (%parent fragment should > 0.7);
#             3. uncharge;
#             4. canonicalize tautomer;
#             5. check atom number < 64;
#             7. elements;

#     """
#     try:
#         with BlockLogs():
#             mol = Chem.MolFromSmiles(smiles, sanitize = False,)
#             clean_mol = rdMolStandardize.Cleanup(mol)
#             parent_clean_mol = rdMolStandardize.FragmentParent(clean_mol)
#             uncharger = rdMolStandardize.Uncharger()
#             uncharged_parent_clean_mol = uncharger.uncharge(parent_clean_mol)
#             te = rdMolStandardize.TautomerEnumerator()  # idem
#             taut_uncharged_parent_clean_mol = te.Canonicalize(uncharged_parent_clean_mol)
#     except:
#         return None
    
#     ### check atom number
#     atom_number = taut_uncharged_parent_clean_mol.GetNumAtoms()
#     original_atom_number = mol.GetNumAtoms()
#     if atom_number > 64:
#         return None
#     if atom_number / original_atom_number < dropout_threshold: ### handmade threshold
#         return None

#     ### check element types
#     valid_types = ['H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'Ge', 'As', 'Br', 'Sb', 'I']
#     element_types = [atom.GetSymbol() for atom in taut_uncharged_parent_clean_mol.GetAtoms()]
#     if set(element_types).issubset(valid_types): ### < 64 atoms
#         canonical_smi = Chem.MolToSmiles(taut_uncharged_parent_clean_mol, isomericSmiles=True, canonical=True)
#         _m = Chem.MolFromSmiles(canonical_smi, sanitize=False)
#         re_canonical_smi = Chem.MolToSmiles(_m, isomericSmiles=True, canonical=True)
#         if canonical_smi == re_canonical_smi:
#             return canonical_smi
#     else:
#         return None

def standardize(smiles, dropout_threshold = 0.7):
    uncharger = rdMolStandardize.Uncharger() 
    try:
        with BlockLogs():
            m_ = Chem.MolFromSmiles(smiles, sanitize = True)
            m_ = uncharger.uncharge(m_)
            cs = Chem.MolToSmiles(m_, isomericSmiles = True, canonical = True)
        return cs
    except:
        return None

def sanitize_smiles(smiles):
    valid_element_set = set([1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 32, 33, 34, 35, 51, 52, 53])
    uncharger = rdMolStandardize.Uncharger() 
    
    try:
        with BlockLogs():
            mol = Chem.MolFromSmiles(smiles, sanitize = True) 
            # mol = rdMolStandardize.Cleanup(mol)
            # mol = rdMolStandardize.FragmentParent(mol)
            mol = uncharger.uncharge(mol)
            # mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
            element_set = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
            ## assert options: 3 < heavy atoms <= 64,
            # print(mol.GetNumHeavyAtoms())
            # print(set(element_set).issubset(valid_element_set))
            assert 3 < mol.GetNumHeavyAtoms() <= 64
            assert set(element_set).issubset(valid_element_set)
            stan_smiles = Chem.MolToSmiles(mol, isomericSmiles = True, canonical = True)
            _m = Chem.MolFromSmiles(stan_smiles, sanitize = True)
            assert _m is not None
            return stan_smiles
    except Exception as e:
        return None

ENCODING_DTYPE = np.int16
def encoding_graphs(smi):

    # debug
    try:
        mol_to_process = Chem.MolFromSmiles(smi, sanitize = True)
        atom_type = [atom.GetAtomicNum() for atom in mol_to_process.GetAtoms()]
    except:
        breakpoint()
    mol_to_process = Chem.MolFromSmiles(smi, sanitize = True)

    # Chem.SanitizeMol(mol_to_process, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES)
    # mol_to_process.UpdatePropertyCache(strict=False)
    # Chem.AssignStereochemistry(mol_to_process, cleanIt=True, force=True)

    #### get atom features:
    #### atom type, formal charge, degree, #H, aromaticity and hybridization, chiral
    atom_type = [atom.GetAtomicNum() for atom in mol_to_process.GetAtoms()]
    atom_type = np.array(atom_type, ENCODING_DTYPE)
    formal_charge = [atom.GetFormalCharge() for atom in mol_to_process.GetAtoms()]
    formal_charge = np.array(formal_charge, ENCODING_DTYPE)
    num_degree = [atom.GetTotalDegree() for atom in mol_to_process.GetAtoms()]
    num_degree = np.array(num_degree, ENCODING_DTYPE)
    num_H = [atom.GetTotalNumHs() for atom in mol_to_process.GetAtoms()]
    num_H = np.array(num_H, ENCODING_DTYPE)
    aromaticity = [atom.GetIsAromatic() for atom in mol_to_process.GetAtoms()]
    aromaticity = np.array(aromaticity, ENCODING_DTYPE)
    hybridization = [atom.GetHybridization() for atom in mol_to_process.GetAtoms()]
    hybridization = np.array(hybridization, ENCODING_DTYPE)
    chiral = []
    chiral_f = {"S": 1, "R": 2}
    for atom in mol_to_process.GetAtoms():
        if atom.HasProp('_CIPCode'):
            try:
                chiral.append(chiral_f[atom.GetProp('_CIPCode')])
            except:
                chiral.append(3) ## unk
        else:
            chiral.append(0) ## others
    chiral = np.array(chiral, ENCODING_DTYPE)

    #### get bond features:
    #### bond type, stereo, conjugated, in ring, graph distance
    num_atoms = mol_to_process.GetNumAtoms()
    bond_type = np.zeros((num_atoms, num_atoms), ENCODING_DTYPE)
    for bond in mol_to_process.GetBonds():
        bond_type[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond.GetBondType()
        bond_type[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond.GetBondType()
    stereo = np.zeros((num_atoms, num_atoms), ENCODING_DTYPE)
    for bond in mol_to_process.GetBonds():
        stereo[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond.GetStereo()
        stereo[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond.GetStereo()
    conjugated = np.zeros((num_atoms, num_atoms), ENCODING_DTYPE)
    for bond in mol_to_process.GetBonds():
        conjugated[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond.GetIsConjugated() + 1
        conjugated[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond.GetIsConjugated() + 1
    in_ring = np.zeros((num_atoms, num_atoms), ENCODING_DTYPE)
    for bond in mol_to_process.GetBonds():
        in_ring[bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()] = bond.IsInRing() + 1
        in_ring[bond.GetEndAtomIdx(), bond.GetBeginAtomIdx()] = bond.IsInRing() + 1 ## for unk
    graph_distance = np.zeros((num_atoms, num_atoms), ENCODING_DTYPE) #### 24-06-19 add graph distance
    distance_matrix = rdmolops.GetDistanceMatrix(mol_to_process)
    graph_distance = np.asarray(distance_matrix, dtype=ENCODING_DTYPE)

    atom_features = {
        "atom_type": atom_type,
        "formal_charge": formal_charge,
        "num_degree": num_degree,
        "num_H": num_H,
        "aromaticity": aromaticity,
        "hybridization": hybridization,
        "chiral": chiral,
        "atom_mask": np.ones_like(atom_type, dtype=ENCODING_DTYPE),
    }

    bond_features = {
        "bond_type": bond_type,
        "stereo": stereo,
        "conjugated": conjugated,
        "in_ring": in_ring,
        "bond_mask": np.array(bond_type > 0, dtype=ENCODING_DTYPE),
        "graph_distance": graph_distance,
    }

    graph_features = {
        # "num_atoms": num_atoms,
        "atom_features": atom_features,
        "bond_features": bond_features,
    }

    return graph_features

def padding_graph(arr: np.ndarray, padding_num: int = 64, dtype: np.dtype = ENCODING_DTYPE):

    arr = np.array(arr, dtype = dtype)
    natom = arr.shape[0]
    npad = padding_num - natom
    if arr.ndim < 2:
        return np.pad(arr, ((0, npad)), mode='constant', constant_values=0)
    elif arr.ndim == 2:
        return np.pad(arr, ((0, npad), (0, npad)), mode='constant', constant_values=0)
    else:
        raise ValueError("The ndim of input array should be less than 2.")

def make_graph_feature(graph_feature, n_padding_atom: int = 64,):
    
    key_dict = {
        'atom_features': list(graph_feature['atom_features'].keys()), 
        'bond_features': list(graph_feature['bond_features'].keys()),
    }
    graph_feature['atom_features'] = {
        k: padding_graph(
            graph_feature['atom_features'][k], n_padding_atom
            ) for k in key_dict['atom_features']
    }
    graph_feature['bond_features'] = {
        k: padding_graph(
            graph_feature['bond_features'][k], n_padding_atom
            ) for k in key_dict['bond_features']
    }
    return graph_feature

def smi2graph_features(smi, n_padding_atom: int = 64):
    graph_feature = encoding_graphs(smi)
    graph_feature = make_graph_feature(graph_feature, n_padding_atom = n_padding_atom)
    return graph_feature