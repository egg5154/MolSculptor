import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))

import jax
import jax.numpy as jnp
import flax.linen as nn

from jax import Array
from ml_collections.config_dict import ConfigDict
from flax.linen.initializers import truncated_normal
from typing import Union, Optional, Tuple

from .encoder import Encoder
from .decoder import Decoder
from src.common.utils import safe_l2_normalize
from src.module.transformer import Transition
from src.common.layers.mlp import MLP

class SeqGenerator(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, input_features, neighbor_list = None):
        
        atom_features = input_features['graph_features']['atom_features']
        bond_features = input_features['graph_features']['bond_features']
        sequence_features = input_features['sequence_features']

        #### get conditional embedding: (B, NQ, F)
        graph_feat = Encoder(
            self.config.encoder, self.global_config
            )(atom_features, bond_features, neighbor_list)

        #### get seq info: (B, K, N)
        seq_tokens = sequence_features['tokens']
        seq_mask = sequence_features['mask']
        seq_rope_index = sequence_features['rope_index']

        #### reshape & broadcast: (B, NQ, F) -> (B*K, NQ, F), (B, K, N) -> (B*K, N)
        batch_size, num_k, _ = seq_tokens.shape
        graph_feat = jnp.expand_dims(graph_feat, axis=1)
        graph_feat = jnp.tile(graph_feat, (1, num_k, 1, 1)).reshape(-1, *graph_feat.shape[-2:])
        seq_tokens, seq_mask, seq_rope_index = jax.tree_util.tree_map(
            lambda f: f.reshape(batch_size*num_k, -1), (seq_tokens, seq_mask, seq_rope_index)
        )

        #### predict logits: (B*K, N, L)
        # print(jax.tree_util.tree_map(jnp.shape, (seq_tokens, seq_mask, seq_rope_index, graph_feat)))
        seq_logits = Decoder(
            self.config.decoder, self.global_config
            )(seq_tokens, seq_mask, seq_rope_index, graph_feat)
        
        return seq_logits

class Projector(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, features):
        #### input features: (B, NQ*D)
        #### -> FFN -> dense (optional) -> L2 norm
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        dropout_flag = self.global_config.dropout_flag
        # features = Transition(
        #     self.config.transition, self.global_config
        # )(features)
        # out_dim = self.config.out_dim
        # if out_dim != features.shape[-1]: ## project to out_dim
        #     features = nn.Dense(
        #         features = out_dim,
        #         dtype = arr_dtype, use_bias = False, param_dtype = jnp.float32,
        #     )(features)
        # features = safe_l2_normalize(features, axis = -1) ## (B, O)
        features = MLP(
            output_sizes = self.config.output_sizes,
            activation = self.config.activation,
            dropout_rate = self.config.dropout_rate,
            dtype = arr_dtype,
            with_bias = False, 
            dropout_flag = dropout_flag,
        )(features)
        features = safe_l2_normalize(features, axis = -1) ## (B, O)
        return features

class L2SeqGenerator(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, input_features, neighbor_list = None):
        
        atom_features = input_features['graph_features']['atom_features']
        bond_features = input_features['graph_features']['bond_features']
        sequence_features = input_features['sequence_features']

        #### get conditional embedding: (B, NQ, F)
        graph_feat = Encoder(
            self.config.encoder, self.global_config
            )(atom_features, bond_features, neighbor_list)
        
        #### project to latent space: (B, NQ, F) -> (B, NQ, D)
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        graph_feat = nn.Dense(
            self.config.latent_dim, kernel_init = truncated_normal(0.01),
            dtype = arr_dtype, use_bias = False, param_dtype = jnp.float32,
        )(graph_feat)
        num_sink_tokens = self.config.num_sink_tokens
        n_ = graph_feat.shape[1]
        graph_feat = graph_feat[:, :n_ - num_sink_tokens] ## remove sink tokens
        graph_feat = safe_l2_normalize(graph_feat, axis = -1) ## l2 norm for every tokens
        # graph_feat *= jnp.sqrt(jnp.float32(self.config.latent_dim)) ## scale by sqrt(d)
        sim_feat = jnp.reshape(graph_feat, (graph_feat.shape[0], -1)) # (B, NQ, D) -> (B, NQ*D)
        sim_feat = Projector(
            self.config.projector, self.global_config)(sim_feat) # (B, NQ*D) -> (B, NQ*D)
        aux = {'graph_feat': graph_feat, 'sim_feat': sim_feat} # (B, NQ, D)
        # aux = {'graph_feat': graph_feat} # (B, NQ, D) ### debug

        #### get seq info: (B, K, N)
        seq_tokens = sequence_features['tokens']
        seq_mask = sequence_features['mask']
        # seq_rope_index = sequence_features['rope_index']
        # breakpoint() ## check here

        #### reshape & broadcast: (B, ...) -> (B*K, ...)
        batch_size, num_k, num_tokens = seq_tokens.shape # (B, K, N)
        seq_rope_index = jnp.arange(num_tokens, dtype=jnp.int32)[None, None, :]
        seq_rope_index = jnp.broadcast_to(seq_rope_index, seq_tokens.shape)
        graph_feat = jnp.expand_dims(graph_feat, axis=1)
        graph_feat = jnp.tile(graph_feat, (1, num_k, 1, 1)).reshape(-1, *graph_feat.shape[-2:])
        seq_tokens, seq_mask, seq_rope_index = jax.tree_util.tree_map(
            lambda f: f.reshape(batch_size*num_k, -1), (seq_tokens, seq_mask, seq_rope_index)
        )

        #### predict logits: (B*K, N, L)
        # print(jax.tree_util.tree_map(jnp.shape, (seq_tokens, seq_mask, seq_rope_index, graph_feat)))
        seq_logits = Decoder(
            self.config.decoder, self.global_config
            )(seq_tokens, seq_mask, seq_rope_index, graph_feat)
        
        return seq_logits, aux

class MMDSeqGenerator(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, input_features, neighbor_list = None):
        
        atom_features = input_features['graph_features']['atom_features']
        bond_features = input_features['graph_features']['bond_features']
        sequence_features = input_features['sequence_features']

        #### get conditional embedding: (B, NQ, F)
        graph_feat = Encoder(
            self.config.encoder, self.global_config
            )(atom_features, bond_features, neighbor_list)
        
        #### project to latent space: (B, NQ, F) -> (B, NQ, D)
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        graph_feat = nn.Dense(
            self.config.latent_dim, kernel_init = truncated_normal(0.01),
            dtype = arr_dtype, use_bias = False, param_dtype = jnp.float32,
        )(graph_feat)
        num_sink_tokens = self.config.num_sink_tokens
        graph_feat = graph_feat[:, :-num_sink_tokens] ## remove sink tokens
        aux = {'graph_feat': graph_feat} # (B, NQ, D)

        #### get seq info: (B, K, N)
        seq_tokens = sequence_features['tokens']
        seq_mask = sequence_features['mask']

        #### reshape & broadcast: (B, ...) -> (B*K, ...)
        batch_size, num_k, num_tokens = seq_tokens.shape # (B, K, N)
        seq_rope_index = jnp.arange(num_tokens, dtype=jnp.int32)[None, None, :]
        seq_rope_index = jnp.broadcast_to(seq_rope_index, seq_tokens.shape)
        graph_feat = jnp.expand_dims(graph_feat, axis=1)
        graph_feat = jnp.tile(graph_feat, (1, num_k, 1, 1)).reshape(-1, *graph_feat.shape[-2:])
        seq_tokens, seq_mask, seq_rope_index = jax.tree_util.tree_map(
            lambda f: f.reshape(batch_size*num_k, -1), (seq_tokens, seq_mask, seq_rope_index)
        )

        #### predict logits: (B*K, N, L)
        #### print(jax.tree_util.tree_map(jnp.shape, (seq_tokens, seq_mask, seq_rope_index, graph_feat)))
        seq_logits = Decoder(
            self.config.decoder, self.global_config
            )(seq_tokens, seq_mask, seq_rope_index, graph_feat)
        
        return seq_logits, aux

class TrainDecoder(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, graph_feat, sequence_features):

        #### get seq info: (B, K, N)
        seq_tokens = sequence_features['tokens']
        seq_mask = sequence_features['mask']

        #### reshape & broadcast: (B, ...) -> (B*K, ...)
        batch_size, num_k, num_tokens = seq_tokens.shape # (B, K, N)
        seq_rope_index = jnp.arange(num_tokens, dtype=jnp.int32)[None, None, :]
        seq_rope_index = jnp.broadcast_to(seq_rope_index, seq_tokens.shape)
        graph_feat = jnp.expand_dims(graph_feat, axis=1)
        graph_feat = jnp.tile(graph_feat, (1, num_k, 1, 1)).reshape(-1, *graph_feat.shape[-2:])
        seq_tokens, seq_mask, seq_rope_index = jax.tree_util.tree_map(
            lambda f: f.reshape(batch_size*num_k, -1), (seq_tokens, seq_mask, seq_rope_index)
        )

        #### predict logits: (B*K, N, L)
        seq_logits = Decoder(
            self.config.decoder, self.global_config
            )(seq_tokens, seq_mask, seq_rope_index, graph_feat)
        
        return seq_logits

