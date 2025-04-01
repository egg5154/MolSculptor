import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))

import jax
import jax.numpy as jnp
import flax.linen as nn

from jax import Array
from ml_collections.config_dict import ConfigDict
from typing import Union, Optional, Tuple

from src.model.transformer import ResiDualTransformerBlock, PostNormTransformerBlock
from src.module.transformer import NormBlock, Transition, OuterProduct
from src.common.utils import gather_neighbor

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
        

class MolCT(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, atom_emb, atom_mask, bond_emb, bond_mask, 
                 neighbor_list = None):

        norm_method = self.global_config.norm_method
        norm_small = self.global_config.norm_small
        sparse_flag = self.global_config.sparse_flag

        def _update(atom_feature, acc_atom_feature, bond_feature, acc_bond_feature):

            #### 1. gather neighbor (if we need)
            if sparse_flag:
                ## (B, A, 1) -> (B, A, A', 1) -> (B, A, A')
                neighbor_mask = gather_neighbor(atom_mask[..., None], neighbor_list, is_pair=False)
                neighbor_mask = jnp.squeeze(neighbor_mask, axis=-1)
            else:
                ## (B, A) -> (B, 1, A)
                neighbor_mask = jnp.expand_dims(atom_mask, axis=-2)

            #### 2. atom feature update
            transformer_config = self.config.node_update
            atom_feature, acc_atom_feature = ResiDualTransformerBlock(
                transformer_config, self.global_config
                )(s_i=atom_feature, acc_s_i=acc_atom_feature, m_i=atom_mask, m_j=neighbor_mask, \
                  z_ij=bond_feature, m_ij=bond_mask, n_i_or_r_i=neighbor_list)
            
            #### 3. gather neighbor (if we need)
            if sparse_flag:
                ## (B, A, Fa) -> (B, A, A', Fa)
                neighbor_atom_feature = gather_neighbor(atom_feature, neighbor_list, is_pair=True)
            else:
                ## (B, A, Fa) -> (B, 1, A, Fa) -> (B, A, A, Fa)
                neighbor_atom_feature = jnp.expand_dims(atom_feature, axis=-3)
                neighbor_atom_feature = jnp.tile(neighbor_atom_feature, (1, atom_feature.shape[-2], 1, 1))
            
            #### 4. bond feature update
            edge_update_config = self.config.edge_update
            bond_feature, acc_bond_feature = PairFeatureUpdate(
                edge_update_config, self.global_config
            )(atom_feature, neighbor_atom_feature, atom_mask, neighbor_mask, bond_feature, acc_bond_feature, bond_mask)
            
            return atom_feature, acc_atom_feature, bond_feature, acc_bond_feature

        n_layers = self.config.n_layers
        atom_feat, acc_atom_feat = atom_emb, atom_emb
        bond_feat, acc_bond_feat = bond_emb, bond_emb
        for _ in range(n_layers):
            atom_feat, acc_atom_feat, bond_feat, acc_bond_feat = _update(
                atom_feat, acc_atom_feat, bond_feat, acc_bond_feat
            )
        acc_atom_feat = NormBlock(norm_method, norm_small)(acc_atom_feat)
        acc_bond_feat = NormBlock(norm_method, norm_small)(acc_bond_feat)
        atom_feat += acc_atom_feat
        bond_feat += acc_bond_feat

        # ## debug
        # print("atom_feat: ", atom_feat[0].max(), atom_feat[0].min())
        # print("bond_feat: ", bond_feat[0].max(), bond_feat[0].min())

        return atom_feat, bond_feat
