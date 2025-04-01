import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))

import jax
import jax.numpy as jnp
import flax.linen as nn

from jax import Array
from ml_collections.config_dict import ConfigDict
from typing import Union, Optional, Tuple
from flax.linen.initializers import zeros_init

from src.module.transformer import NormBlock, Transition
from src.module.attention import AttentionEmbedding, HyperAttentionEmbedding, AttentionKernel, PostAttention
from src.model.transformer import ResiDualTransformerBlock, PostNormTransformerBlock
from src.common.utils import get_activation

class adaLN(nn.Module):

    hidden_size: int
    global_config: ConfigDict
    module: nn.Module
    activation: str = 'silu'

    @nn.compact
    def __call__(self, x, cond, other_inputs = ()):
        #### Input: x: (B, ..., F) [CURRENTLY: (B, T, F)], cond: (B, F)

        #### 1. generate alpha, gamma, beta
        hidden_size = self.hidden_size
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        cond = get_activation(self.activation)(cond)
        cond = nn.Dense(
            features = 3 * hidden_size,
            kernel_init = zeros_init(),
            dtype = arr_dtype,
            param_dtype = jnp.float32,
        )(cond) # (B, 3 * F)
        alpha, beta, gamma = jnp.split(cond, 3, -1) # (B, F)

        #### 2. main function
        norm_small = self.global_config.norm_small
        act, d_act = x, x
        d_act = NormBlock(eps = norm_small)(d_act)
        d_act = d_act * (1 + gamma[:, None]) + beta[:, None]
        d_act = self.module(d_act, *other_inputs)
        d_act = d_act * alpha[:, None]
        act += d_act

        return act

#### residual adaLN
class adaLNRes(nn.Module):

    config: ConfigDict
    # hidden_size: int
    global_config: ConfigDict
    module: nn.Module
    # activation: str = 'silu'

    @nn.compact
    def __call__(self, x, acc_x, cond, other_inputs = ()):
        #### Input: x: (B, ..., F) [CURRENTLY: (B, T, F)], cond: (B, F)

        #### 1. generate alpha, gamma, beta
        hidden_size = self.config.hidden_size
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        cond = get_activation(self.config.activation)(cond)
        cond = nn.Dense(
            features = 3 * hidden_size,
            kernel_init = zeros_init(),
            dtype = arr_dtype,
            param_dtype = jnp.float32,
        )(cond) # (B, 3 * F)
        alpha, beta, gamma = jnp.split(cond, 3, -1) # (B, F)

        #### 2. main function
        norm_eps = self.global_config.norm_small
        x, d_x = x, x
        d_x = d_x * (1 + gamma[:, None]) + beta[:, None]
        d_x = self.module(d_x, *other_inputs)
        d_x = d_x * alpha[:, None]
        x += d_x
        acc_x += d_x
        x = NormBlock(eps = norm_eps, norm_method = self.config.norm_method)(x)

        return x, acc_x

class AttentionBlock(nn.Module):

    config: ConfigDict
    global_config: ConfigDict
    hyper_lora_config: Optional[ConfigDict]

    @nn.compact
    def __call__(self, single_act, single_mask, rope_index = None, hyper_var = None):

        #### 1. Attention Embedding
        embedding_config = self.config.attention_embedding
        q, k, v, _ = AttentionEmbedding(
            embedding_config, self.global_config, self.hyper_lora_config
        )(single_act, hyper_var = hyper_var)

        #### 2. HyperAttention Embedding
        if self.config.hyper_attention_flag:
            hyper_embedding_config = self.config.hyper_attention_embedding
            q, k = HyperAttentionEmbedding(
                hyper_embedding_config, self.global_config
            )(q, k, None, None, None, rope_index)
        
        #### 3. Attention Kernel
        kernel_config = self.config.attention_kernel
        out_act = AttentionKernel(kernel_config, self.global_config)(q, k, v, None, single_mask)

        #### 4. Post Attention
        post_attention_config = self.config.post_attention
        out_act = PostAttention(
            post_attention_config, self.global_config, self.hyper_lora_config,
        )(out_act, q, hyper_var = hyper_var)

        #### 5. dropout
        dropout_flag = self.global_config.dropout_flag
        out_act = nn.Dropout(
            rate=self.config.dropout_rate, deterministic=(not dropout_flag)
            )(out_act)

        return out_act

class TransitionBlock(nn.Module):

    config: ConfigDict
    global_config: ConfigDict
    hyper_lora_config: Optional[ConfigDict]

    @nn.compact
    def __call__(self, act, hyper_var = None) -> Array:
        
        #### 1. Transition (GLU or FFN)
        act = Transition(
            self.config.transition, self.global_config, self.hyper_lora_config,
        )(act, hyper_var = hyper_var)

        #### 2. Dropout
        dropout_flag = self.global_config.dropout_flag
        act = nn.Dropout(
            rate=self.config.dropout_rate, deterministic=(not dropout_flag)
            )(act)

        return act

class adaLNPreNormTransformerBlock(nn.Module):

    config: ConfigDict
    global_config: ConfigDict
    hyper_lora_config: Optional[ConfigDict]

    @nn.compact
    def __call__(self, tokens, tokens_mask, tokens_rope_index, cond):
        ### Inputs: tokens: (B, T, F), cond: (B, F)
        ### Returns: act: (B, T, F)

        act = tokens
        hyper_var = cond if self.hyper_lora_config else None
        #### 1. Attention
        attention_block = AttentionBlock(self.config.attention, self.global_config, self.hyper_lora_config)
        add_info = (tokens_mask, tokens_rope_index, hyper_var) ### should be in order
        act = adaLN(
            **self.config.adaLN, global_config=self.global_config, module=attention_block)(act, cond, add_info)

        #### 2. Transition
        transition_block = TransitionBlock(self.config.transition, self.global_config, self.hyper_lora_config)
        add_info = (hyper_var,)
        act = adaLN(
            **self.config.adaLN, global_config=self.global_config, module=transition_block)(act, cond, add_info)
        
        return act

class adaLNResTransformerBlock(nn.Module):

    config: ConfigDict
    global_config: ConfigDict
    hyper_lora_config: Optional[ConfigDict]

    @nn.compact
    def __call__(self, tokens, acc_tokens, tokens_mask, tokens_rope_index, cond):
        ### Inputs: tokens: (B, T, F), cond: (B, F)
        ### Returns: act: (B, T, F)

        hyper_var = cond if self.hyper_lora_config else None
        #### 1. Attention
        attention_block = AttentionBlock(self.config.attention, self.global_config, self.hyper_lora_config)
        add_info = (tokens_mask, tokens_rope_index, hyper_var) ### should be in order
        tokens, acc_tokens = adaLNRes(
            self.config.adaLN, global_config=self.global_config, module=attention_block
            )(tokens, acc_tokens, cond, add_info)

        #### 2. Transition
        transition_block = TransitionBlock(self.config.transition, self.global_config, self.hyper_lora_config)
        add_info = (hyper_var,)
        tokens, acc_tokens = adaLNRes(
            self.config.adaLN, global_config=self.global_config, module=transition_block
            )(tokens, acc_tokens, cond, add_info)
        
        return tokens, acc_tokens

class Decoder(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, seq_tokens: Array, mask: Array, 
                 rope_index: Array, cond: Array):
        
        norm_small = self.global_config.norm_small
        # norm_method = self.global_config.norm_method
        norm_method = self.config.norm_method
        _dtype = jnp.float32 if self.global_config.bf16_flag == False else jnp.bfloat16

        transformer_config = self.config.transformer
        n_layers = transformer_config.n_layers
        num_prefix_tokens = self.config.num_prefix_tokens
        n_groups = self.config.num_groups
        assert n_layers % n_groups == 0
        assert num_prefix_tokens % n_groups == 0
        n_layers_per_group = n_layers // n_groups
        num_tpg = num_prefix_tokens // n_groups

        hyper_lora_flag = self.config.hyper_lora_flag
        hyper_lora_config = self.config.lora_config if hyper_lora_flag else None
        # hyper_var_ = {}
        
        num_seq_tokens = self.config.settings_config.vocab_size
        logits_dim = num_seq_tokens

        #### embedding sequence
        dim_feature = self.config.dim_feature
        seq_emb = nn.Embed(
            num_embeddings=num_seq_tokens + 3, ## BOS, EOS, UNK
            features=dim_feature,
            dtype=_dtype,
            param_dtype=jnp.float32,
        )(seq_tokens)

        # #### adaLN prenorm transformer block ####
        # prefix_emb = cond
        # ## (B, N_PRE, ...) | (B, N, ...) -> (B, N_L + N, ...)
        # graph_seq_emb = jnp.concatenate([prefix_emb, seq_emb], axis=1)
        # batch_size = graph_seq_emb.shape[0]
        # prefix_rope_index = jnp.repeat(
        #         jnp.arange(
        #             -100 - num_prefix_tokens, 
        #             -100, 
        #             dtype=rope_index.dtype
        #             )[None, ...], 
        #         batch_size, axis=0,
        #     )
        # rope_index = jnp.concatenate([prefix_rope_index, rope_index], axis=1)
        # rope_index = rope_index[:, :-num_prefix_tokens]
        # emb = graph_seq_emb[:, :-num_prefix_tokens]
        # for _g in range(n_groups):
        #     # (B, D_N_PRE*N_GROUP,) | (B, N_PRE - D_N_PRE*N_GROUP) | (B, N_L, F)
        #     mask = jnp.concatenate([
        #         jnp.ones(
        #             shape=(batch_size, num_tpg * (_g + 1)), dtype=mask.dtype),
        #         jnp.zeros(
        #             shape=(batch_size, num_tpg * (n_groups - (_g + 1))), dtype=mask.dtype),
        #         mask,
        #     ], axis=1)[:, :-num_prefix_tokens]
        #     # make hyper var: (B, D_N_PRE, F) -> (B, D_N_PRE*F)
        #     hyper_var_emb = cond[:, num_tpg * _g:num_tpg * (_g + 1)]
        #     hyper_var_ = hyper_var_emb.reshape(batch_size, -1)
        #     for _ in range(n_layers_per_group):
        #         emb = adaLNPreNormTransformerBlock(
        #             self.config.transformer, self.global_config, hyper_lora_config,
        #         )(emb, mask, rope_index, cond = hyper_var_)
        # emb = NormBlock(norm_method, norm_small)(emb)

        #### adaLN prenorm transformer block ####
        ## (B, N_PRE, D) -> (B, N_PRE, F)
        prefix_emb = nn.Dense(
            dim_feature, dtype = _dtype,
            param_dtype = jnp.float32, use_bias = False,
        )(cond)
        ## (B, N_PRE, ...) | (B, N, ...) -> (B, N_L + N, ...)
        graph_seq_emb = jnp.concatenate([prefix_emb, seq_emb], axis=1)
        batch_size = graph_seq_emb.shape[0]
        prefix_rope_index = jnp.repeat(
                jnp.arange(
                    -100 - num_prefix_tokens, 
                    -100, 
                    dtype=rope_index.dtype
                    )[None, ...], 
                batch_size, axis=0,
            )
        rope_index = jnp.concatenate([prefix_rope_index, rope_index], axis=1)
        rope_index = rope_index[:, :-num_prefix_tokens]
        emb = graph_seq_emb[:, :-num_prefix_tokens]
        emb, acc_emb = emb, emb
        for _g in range(n_groups):
            # (B, D_N_PRE*N_GROUP,) | (B, N_PRE - D_N_PRE*N_GROUP) | (B, N_L, F)
            mask = jnp.concatenate([
                jnp.ones(
                    shape=(batch_size, num_tpg * (_g + 1)), dtype=mask.dtype),
                jnp.zeros(
                    shape=(batch_size, num_tpg * (n_groups - (_g + 1))), dtype=mask.dtype),
                mask,
            ], axis=1)[:, :-num_prefix_tokens]
            # make hyper var: (B, D_N_PRE, F) -> (B, D_N_PRE*F)
            hyper_var_emb = cond[:, num_tpg * _g:num_tpg * (_g + 1)]
            hyper_var_ = hyper_var_emb.reshape(batch_size, -1)
            for _ in range(n_layers_per_group):
                emb, acc_emb = adaLNResTransformerBlock(
                    self.config.transformer, self.global_config, hyper_lora_config,
                )(emb, acc_emb, mask, rope_index, cond = hyper_var_)
        emb += NormBlock(norm_method, norm_small)(acc_emb)
        
        #### post process
        ## (B, N, F) -> (B, N, L)
        logits = nn.Dense(features=logits_dim, kernel_init=zeros_init(), use_bias=True, dtype=_dtype)(emb)

        return logits

        


        

