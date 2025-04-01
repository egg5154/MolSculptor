import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))

import jax
import jax.numpy as jnp
import flax.linen as nn

from jax import Array
from jax.debug import breakpoint
from flax.linen.initializers import variance_scaling
from flax.linen.dtypes import promote_dtype
from ml_collections.config_dict import ConfigDict
from typing import Union, Optional, Tuple

from net.embedding import EmbeddingBlock
from net.molct_plus import MOLCT_PLUS
from src.module.transformer import NormBlock

default_embed_init = variance_scaling(
  1.0, 'fan_in', 'normal', out_axis=0
)

class Encoder(nn.Module):

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
        graph_distance = jax.nn.one_hot(
            bond_features['graph_distance'], num_classes = bond_emb_config.n_graph_distance,
            dtype = arr_dtype,
        )
        ## (B, A, A, C'_1) | ... | (B, A, A, C'_n) -> (B, A, A, C')
        bond_raw_feat = jnp.concatenate(
            [bond_type, stereo, conjugated, in_ring, graph_distance],
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
        
        
