import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))

import jax
import jax.numpy as jnp
import flax.linen as nn

from jax import Array
from ml_collections.config_dict import ConfigDict
from typing import Union, Optional, Tuple

class EmbeddingBlock(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, atom_features, bond_features,):

        _dtype = jnp.float32 if self.global_config.bf16_flag == False else jnp.bfloat16
        dim_atom_feature = self.config.dim_atom_feature
        dim_bond_feature = self.config.dim_bond_feature ## atom feature and bond feature are the same.
        embedding_config = self.config.embedding
        
        #### extract atom features
        atom_emb_config = embedding_config.atom_embedding
        
        atom_type = atom_features["atom_type"]
        formal_charge = atom_features["formal_charge"]
        num_H = atom_features["num_H"]
        aromaticity = atom_features["aromaticity"]
        hybridization = atom_features["hybridization"]
        chiral = atom_features["chiral"]

        atom_type_emb = nn.Embed(num_embeddings=atom_emb_config.n_atom_type,
                                 features=dim_atom_feature,
                                 dtype=_dtype,
                                 param_dtype=jnp.float32,)(atom_type)
        formal_charge_emb = nn.Embed(num_embeddings=atom_emb_config.n_formal_charge,
                                     features=dim_atom_feature,
                                     dtype=_dtype,
                                     param_dtype=jnp.float32,)(formal_charge + 1)
        num_H_emb = nn.Embed(num_embeddings=atom_emb_config.n_num_H,
                             features=dim_atom_feature,
                             dtype=_dtype,
                             param_dtype=jnp.float32,)(num_H)
        aromaticity_emb = nn.Embed(num_embeddings=atom_emb_config.n_aromaticity,
                                   features=dim_atom_feature,
                                   dtype=_dtype,
                                   param_dtype=jnp.float32,)(aromaticity)
        hybridization_emb = nn.Embed(num_embeddings=atom_emb_config.n_hybridization,
                                     features=dim_atom_feature,
                                     dtype=_dtype,
                                     param_dtype=jnp.float32,)(hybridization)
        chiral_emb = nn.Embed(num_embeddings=atom_emb_config.n_chiral,
                              features=dim_atom_feature,
                              dtype=_dtype,
                              param_dtype=jnp.float32,)(chiral)
        atom_emb = atom_type_emb + formal_charge_emb + num_H_emb + aromaticity_emb + \
            hybridization_emb + chiral_emb
        
        #### extract bond features
        bond_emb_config = embedding_config.bond_embedding

        bond_type = bond_features["bond_type"]
        stereo = bond_features["stereo"]
        conjugated = bond_features["conjugated"]
        in_ring = bond_features["in_ring"]
        graph_distance = bond_features["graph_distance"]

        bond_type_emb = nn.Embed(num_embeddings=bond_emb_config.n_bond_type,
                                 features=dim_bond_feature,
                                 dtype=_dtype,
                                 param_dtype=jnp.float32,)(bond_type)
        stereo_emb = nn.Embed(num_embeddings=bond_emb_config.n_stereo,
                              features=dim_bond_feature,
                              dtype=_dtype,
                              param_dtype=jnp.float32,)(stereo)
        conjugated_emb = nn.Embed(num_embeddings=bond_emb_config.n_conjugated,
                                  features=dim_bond_feature,
                                  dtype=_dtype,
                                  param_dtype=jnp.float32,)(conjugated)
        in_ring_emb = nn.Embed(num_embeddings=bond_emb_config.n_in_ring,
                               features=dim_bond_feature,
                               dtype=_dtype,
                               param_dtype=jnp.float32,)(in_ring)
        graph_distance_emb = nn.Embed(num_embeddings=bond_emb_config.n_graph_distance,
                                      features=dim_bond_feature,
                                      dtype=_dtype,
                                      param_dtype=jnp.float32,)(graph_distance)

        bond_emb = bond_type_emb + stereo_emb + conjugated_emb + in_ring_emb \
              + graph_distance_emb

        return atom_emb, bond_emb