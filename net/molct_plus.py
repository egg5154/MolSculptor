"""
Modified MOLCT-PLUS with Prefix Transformer.
24-07-22
Compare with Classical MOLCT-PLUS:
    1. Use ResiDual Transformer's connection.
    2. Use HAK (not HyperAttention).
"""

import jax
import math
import flax.linen as nn
import jax.numpy as jnp

from ml_collections import ConfigDict
from flax.linen.initializers import lecun_normal

from src.model.transformer import ResiDualTransformerBlock
from src.module.transformer import NormBlock, Transition, OuterProduct
from src.common.utils import gather_neighbor

class MOLCT_PLUS(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, atom_raw_feat, atom_mask, 
                 pair_raw_feat, pair_mask, neighbor_list = None):
        r"""MolCT+ by ZhangJ.
        Args:
            atom_raw_feat: [B, A, C];
            atom_mask: [B, A,];
            pair_raw_feat: [B, A, A, C'];
            pair_mask: [B, A, A];
            neighbor_list: [B, A, A'];
        Return:
            mol_act: [F,]
            atom_feat: [A, F]
            bond_feat: [A, A, F']
        """

        norm_method = self.global_config.norm_method
        norm_eps = self.global_config.norm_small

        ### Interaction
        atom_feat, pair_feat = 0, 0
        acc_atom_feat, acc_pair_feat = FeatureTransformer(
            self.config.feat_transformer, self.global_config,
        )(atom_raw_feat, pair_raw_feat)
        for _ in range(self.config.n_layers):

            ## 1. Re-initialize Features
            d_atom_act, d_pair_act = FeatureTransformer(
                self.config.feat_transformer, self.global_config,
            )(atom_raw_feat, pair_raw_feat)
            atom_feat += d_atom_act
            pair_feat += d_pair_act
            
            ## 2) Run NIU:
            # (A, Cm), (A, A, Cz):
            atom_feat, acc_atom_feat, pair_feat, acc_pair_feat = InteractionUnit(
                self.config.graph_transformer, self.global_config,
            )(atom_feat, acc_atom_feat, atom_mask, pair_feat, acc_pair_feat, pair_mask, neighbor_list)

        acc_atom_feat = NormBlock(norm_method, norm_eps)(acc_atom_feat)
        acc_pair_feat = NormBlock(norm_method, norm_eps)(acc_pair_feat)
        atom_feat += acc_atom_feat
        pair_feat += acc_pair_feat
        
        return atom_feat, pair_feat

class FeatureTransformer(nn.Module):
    r"""Perform Raw Feature Embedding."""

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, atom_raw_feat, pair_raw_feat):
        r"""
        Inputs:
            atom_raw_feat: one-hot encoding for atom, shape of (B, A, C)
            pair_raw_feat: one-hot encoding for bond, shape of (B, A, A, C')
        """

        ### configs
        _dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        atom_act_dim = self.config.atom_act_dim
        pair_act_dim = self.config.pair_act_dim

        # (..., A, C) -> (..., A, F):
        atom_act = nn.Dense(
            features = atom_act_dim, dtype = _dtype,
            param_dtype = jnp.float32, kernel_init = lecun_normal(),
            name = 'preprocess_1d',
        )(atom_raw_feat)        
        # (..., A, A, C') -> (..., A, A, F'):
        pair_act = nn.Dense(
            features = pair_act_dim, dtype = _dtype, param_dtype = jnp.float32,
            kernel_init = lecun_normal(), name = 'pair_activations',
        )(pair_raw_feat)
        # (..., A, C) -> (..., A, F'):
        left_act = nn.Dense(
            features = pair_act_dim, dtype = _dtype, param_dtype = jnp.float32,
            kernel_init = lecun_normal(), name = 'left_single',
        )(atom_raw_feat)
        right_act = nn.Dense(
            features = pair_act_dim, dtype = _dtype, param_dtype = jnp.float32,
            kernel_init = lecun_normal(), name = 'right_single',
        )(atom_raw_feat)

        # (..., A, 1, F) + (..., 1, A, F) -> (..., A, A, F)
        pair_act += jnp.expand_dims(left_act, -2) + jnp.expand_dims(right_act, -3)

        return atom_act, pair_act

class PairFeatureUpdate(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, s_i, s_j, m_i, m_j, z_ij, acc_z_ij, m_ij):

        #### set global config
        norm_method = self.global_config.norm_method
        norm_small = self.global_config.norm_small
        dropout_flag = self.global_config.dropout_flag

        #### 1. Outer Product
        acc_act = acc_z_ij
        act, d_act = z_ij, jnp.zeros_like(z_ij)
        d_act = OuterProduct(
            self.config.outer_product, self.global_config,
            )(s_i, s_j, m_i, m_j)
        d_act = nn.Dropout(
            rate=self.config.dropout_rate, deterministic=(not dropout_flag)
            )(d_act)
        act += d_act
        acc_act += d_act
        act = NormBlock(norm_method, norm_small)(act)

        #### 2. Transition
        act, d_act = act, act
        d_act = Transition(
            self.config.transition, self.global_config
            )(d_act)
        d_act = nn.Dropout(
            rate=self.config.dropout_rate, deterministic=(not dropout_flag)
            )(d_act)
        act += d_act
        acc_act += d_act
        act = NormBlock(norm_method, norm_small)(act)

        #### 3. Mask
        act = act * m_ij[..., None]
        acc_act = acc_act * m_ij[..., None]

        return act, acc_act

class InteractionUnit(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, atom_act, acc_atom_act, atom_mask, 
                 pair_act, acc_pair_act, pair_mask, neighbor_list = None):
        
        sparse_flag = self.global_config.sparse_flag

        def _atom_update(atom_feat, acc_atom_feat, atom_mask, 
                         pair_feat, pair_mask, neighbor_list):

            #### 1. gather neighbor (if we need)
            if sparse_flag:
                ## (B, A, 1) -> (B, A, A', 1) -> (B, A, A')
                neighbor_mask = gather_neighbor(atom_mask[..., None], neighbor_list, is_pair=False)
                neighbor_mask = jnp.squeeze(neighbor_mask, axis=-1)
            else:
                ## (B, A) -> (B, 1, A)
                neighbor_mask = jnp.expand_dims(atom_mask, axis=-2)
            
            #### 2. atom feature update
            atom_feat, acc_atom_feat = ResiDualTransformerBlock(
                self.config.node_update, self.global_config,
            )(s_i = atom_feat, acc_s_i = acc_atom_feat, m_i = atom_mask, m_j = neighbor_mask, \
              z_ij = pair_feat, m_ij = pair_mask, n_i_or_r_i = neighbor_list)
            
            return atom_feat, acc_atom_feat, neighbor_mask

        def _pair_update(pair_feat, acc_pair_feat, pair_mask,
                         atom_feat, atom_mask, neighbor_list, neighbor_mask,):

            #### 1. gather neighbor (if we need)
            if sparse_flag:
                ## (B, A, Fa) -> (B, A, A', Fa)
                neighbor_atom_feat = gather_neighbor(atom_feat, neighbor_list, is_pair=True)
            else:
                ## (B, A, Fa) -> (B, 1, A, Fa) -> (B, A, A, Fa)
                neighbor_atom_feat = jnp.expand_dims(atom_feat, axis=-3)
                neighbor_atom_feat = jnp.tile(neighbor_atom_feat, (1, atom_feat.shape[-2], 1, 1))
            
            #### 2. bond feature update
            pair_feat, acc_pair_feat = PairFeatureUpdate(
                self.config.edge_update, self.global_config
            )(atom_feat, neighbor_atom_feat, atom_mask, neighbor_mask, \
              pair_feat, acc_pair_feat, pair_mask)
            
            return pair_feat, acc_pair_feat
        
        ### Update Pair representation
        atom_act, acc_atom_act, neighbor_mask = _atom_update(
            atom_act, acc_atom_act, atom_mask, pair_act, pair_mask, neighbor_list)

        ### Update Atom Representation
        for _ in range(self.config.n_pair_interactions):
            pair_act, acc_pair_act = _pair_update(
                pair_act, acc_pair_act, pair_mask, atom_act, atom_mask, neighbor_list, neighbor_mask)
            
        return atom_act, acc_atom_act, pair_act, acc_pair_act

class PrefixGraphTransformerPLUS(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    def create_raw_feature(self, atom_features, bond_features):
        """create one-hot feature for MOLCT-PLUS"""

        embedding_config = self.config.embedding
        atom_emb_config = embedding_config.atom_embedding
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        ##### atom feature embedding: (B, A, C_i)
        atom_type = jax.nn.one_hot(
            atom_features['atom_type'], num_classes = atom_emb_config.n_atom_type,
            dtype = arr_dtype,
        )
        formal_charge = jax.nn.one_hot(
            atom_features['formal_charge'], num_classes = atom_emb_config.n_formal_charge,
            dtype = arr_dtype,
        )
        num_H = jax.nn.one_hot(
            atom_features['num_H'], num_classes = atom_emb_config.n_num_H,
            dtype = arr_dtype,
        )
        aromaticity = jax.nn.one_hot(
            atom_features['aromaticity'], num_classes = atom_emb_config.n_aromaticity,
            dtype = arr_dtype,
        )
        hybridization = jax.nn.one_hot(
            atom_features['hybridization'], num_classes = atom_emb_config.n_hybridization,
            dtype = arr_dtype,
        )
        chirality = jax.nn.one_hot(
            atom_features['chiral'], num_classes = atom_emb_config.n_chiral,
            dtype = arr_dtype,
        )
        ## (B, A, C_1) | ... | (B, A, C_n) -> (B, A, C)
        atom_raw_feat = jnp.concatenate(
            [atom_type, formal_charge, num_H, aromaticity, hybridization, chirality],
            axis = -1,
        )
        ## (B, A, C) -> (B, A, C + num_prefix) for prefix atoms
        bs, a_, c_ = atom_raw_feat.shape
        n_p = self.config.num_prefix_atoms
        atom_raw_feat = jnp.concatenate(
            [atom_raw_feat, jnp.zeros((bs, a_, n_p), dtype = jnp.int32)], axis = -1,
        )
        ## (B, A, C + num_prefix) | (B, num_prefix, C + num_prefix)
        prefix_raw_feat = jnp.eye(n_p, dtype = jnp.int32) # (num_prefix, num_prefix)
        prefix_raw_feat = jnp.pad(
            prefix_raw_feat, ((0, 0), (c_, 0)), mode = 'constant',
            constant_values = 0,
        ) # (num_prefix, C + num_prefix)
        prefix_raw_feat = jnp.repeat(
            prefix_raw_feat[None, ...], repeats = bs, axis = 0,
        ) # (B, num_prefix, C + num_prefix)
        atom_raw_feat = jnp.concatenate(
            [prefix_raw_feat, atom_raw_feat], axis = 1,
        )
        atom_mask = jnp.pad(
            atom_features['atom_mask'], ((0, 0), (n_p, 0)),
            mode = 'constant', constant_values = 1,
        )

        ##### bond feature embedding
        bond_emb_config = embedding_config.bond_embedding
        bond_type = jax.nn.one_hot(
            bond_features['bond_type'], num_classes = bond_emb_config.n_bond_type,
            dtype = arr_dtype,
        )
        stereo = jax.nn.one_hot(
            bond_features['stereo'], num_classes = bond_emb_config.n_stereo,
            dtype = arr_dtype,
        )
        conjugated = jax.nn.one_hot(
            bond_features['conjugated'], num_classes = bond_emb_config.n_conjugated,
            dtype = arr_dtype,
        )
        in_ring = jax.nn.one_hot(
            bond_features['in_ring'], num_classes = bond_emb_config.n_in_ring,
            dtype = arr_dtype,
        )
        ## (B, A, A, C'_1) | ... | (B, A, A, C'_n) -> (B, A, A, C')
        bond_raw_feat = jnp.concatenate(
            [bond_type, stereo, conjugated, in_ring],
            axis = -1,
        )
        # (B, A, A, C') -> (B, A, A, C' + 1) -> (B, A + num_prefix, A + num_prefix, C' + 1)
        c_b = bond_raw_feat.shape[-1]
        bond_raw_feat = jnp.concatenate(
            [bond_raw_feat, jnp.zeros((bs, a_, a_, 1), dtype = jnp.int32)], axis = -1,
        )
        bond_raw_feat = jnp.pad(
            bond_raw_feat, ((0, 0), (n_p, 0), (n_p, 0), (0, 0)),
            mode = 'constant', constant_values = 0,
        )
        # (B, A, A, C' + 1) -> (B, A + num_prefix, A + num_prefix, C' + 1)
        prefix_bond_mask = jnp.pad(
            jnp.zeros_like(bond_features['bond_mask'], dtype = jnp.int32),
            ((0, 0), (n_p, 0), (n_p, 0)),
            mode = 'constant', constant_values = 1,
        )
        prefix_bond_raw_feat = jnp.concatenate([
            jnp.zeros((c_b,), dtype = jnp.int32),
            jnp.ones((1,), dtype = jnp.int32),],
            axis = 0,
        )[None, None, None, ...] # (1, 1, 1, C' + 1)
        bond_raw_feat = jnp.where(
            prefix_bond_mask[..., None], prefix_bond_raw_feat, bond_raw_feat,
        ) # (B, A + num_prefix, A + num_prefix, C' + 1)
        bond_mask = jnp.pad(
            bond_features['bond_mask'], ((0, 0), (n_p, 0), (n_p, 0)),
            mode = 'constant', constant_values = 1,
        )

        return atom_raw_feat, atom_mask, bond_raw_feat, bond_mask

    @nn.compact
    def __call__(self, atom_features, bond_features, neighbor_list = None):
        
        num_prefix_atoms = self.config.num_prefix_atoms
        #### 1. create raw feature
        atom_raw_feat, atom_mask, bond_raw_feat, bond_mask = \
            self.create_raw_feature(atom_features, bond_features,)

        #### 4. graph embedding
        molct_config = self.config.molct
        atom_feat, bond_feat = MOLCT_PLUS(
            molct_config, self.global_config,
            )(atom_raw_feat, atom_mask, bond_raw_feat, bond_mask, neighbor_list)

        #### 5. return prefix atom embedding
        prefix_atom_feat = atom_feat[:, :num_prefix_atoms]
        return prefix_atom_feat