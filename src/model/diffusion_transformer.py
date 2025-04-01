import jax
import math
import flax.linen as nn
import jax.numpy as jnp

from typing import Any, Tuple
from flax.linen.initializers import normal, xavier_uniform, zeros_init
from jax import lax
from ml_collections import ConfigDict
from ..module.transformer import NormBlock
from ..common.utils import get_activation
from .transformer import AttentionBlock, TransitionBlock

###### Diffusion Transformer Model from: https://github.com/kvfrans/jax-diffusion-transformer

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, t):

        hidden_size = self.config.hidden_size
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        x = self.timestep_embedding(t)
        x = nn.Dense(hidden_size, kernel_init=normal(0.02), dtype=arr_dtype)(x)
        x = nn.silu(x)
        x = nn.Dense(hidden_size, kernel_init=normal(0.02), dtype=arr_dtype)(x)
        return x

    # t is between [0, max_period]. It's the INTEGER timestep, not the fractional (0,1).;
    def timestep_embedding(self, t, max_period = 10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py

        t = jax.lax.convert_element_type(t, jnp.float32)
        dim = self.config.frequency_embedding_size
        half = dim // 2
        freqs = jnp.exp(-math.log(max_period) * jnp.arange(start=0, stop=half, dtype=jnp.float32) / half)
        args = t[:, None] * freqs[None]
        embedding = jnp.concatenate([jnp.cos(args), jnp.sin(args)], axis=-1) ### TODO: pi here?
        return embedding

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """

    config: ConfigDict
    global_config: ConfigDict

    def token_drop(self, labels, force_drop_ids = None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            rng = self.make_rng('label_dropout')
            batch_size = labels.shape[0]
            drop_ids = jax.random.bernoulli(rng, self.config.label_drop_rate, (batch_size,))
        else:
            drop_ids = force_drop_ids == 1
        labels = jnp.where(drop_ids, self.config.num_classes, labels)
        return labels
    
    @nn.compact
    def __call__(self, labels, force_drop_ids = None):
        ### labels: (B,)

        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        embedding_table = nn.Embed(
            self.config.num_classes + 1, 
            self.config.hidden_size, 
            embedding_init = normal(0.02),
            dtype = arr_dtype,
            )

        use_dropout = self.config.label_drop_flag
        if use_dropout:
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = embedding_table(labels)

        return embeddings

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
            kernel_init = zeros_init() if (not self.global_config.test_flag) else xavier_uniform(),
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

class DiffusionTransformerBlock(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, tokens, tokens_mask, tokens_rope_index, cond):
        ### Inputs: tokens: (B, T, F), cond: (B, F)
        ### Returns: act: (B, T, F)

        act = tokens
        #### 1. Attention
        attention_block = AttentionBlock(self.config.attention, self.global_config)
        add_info = (tokens_mask, tokens_rope_index) ### should be in order
        act = adaLN(
            **self.config.adaLN, global_config=self.global_config, module=attention_block)(act, cond, add_info)

        #### 2. Transition
        transition_block = TransitionBlock(self.config.transition, self.global_config)
        act = adaLN(
            **self.config.adaLN, global_config=self.global_config, module=transition_block)(act, cond)
        
        return act

class DiffusionTransformerOutput(nn.Module):

    hidden_size: int
    output_size: int
    global_config: ConfigDict
    activation: str = 'silu'

    @nn.compact
    def __call__(self, tokens, cond):
        
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        cond = get_activation(self.activation)(cond)
        cond = nn.Dense(
            features = self.hidden_size * 2,
            kernel_init = zeros_init() if (not self.global_config.test_flag) else xavier_uniform(),
            dtype = arr_dtype,
            param_dtype = jnp.float32,
        )(cond)
        beta, gamma = jnp.split(cond, 2, -1)

        act = tokens
        act = NormBlock(eps = self.global_config.norm_small)(act)
        act = act * (1 + gamma[:, None]) + beta[:, None]
        act = nn.Dense(
            features = self.output_size,
            kernel_init = zeros_init() if (not self.global_config.test_flag) else xavier_uniform(),
            dtype = arr_dtype,
            param_dtype = jnp.float32,
        )(act)

        return act

TimeEmbedding = TimestepEmbedder ### TODO: time embedding need to be checked
LabelEmbedding = LabelEmbedder
class DiffusionTransformer(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, tokens, tokens_mask, time, label = None, 
                 force_drop_ids = None, tokens_rope_index = None):
        """
        Inputs:
            tokens: (B, T, C/F)
            tokens_mask: (B, T)
            time: (B,)
            label: (B,)
            force_drop_ids: (B, T)
            tokens_rope_index: (B, T)
        Returns:
            tokens: (B, ...)
        """

        batch_size, n_tokens, channel_size = tokens.shape
        time_emb = TimeEmbedding(self.config.time_embedding, self.global_config)(time) # (B, F)
        if self.config.emb_label_flag:
            label_emb = LabelEmbedding(self.config.label_embedding, self.global_config)(label, force_drop_ids) # (B, F)
        else:
            label_emb = 0.
        condition_emb = time_emb + label_emb

        raw_emb_dim = tokens.shape[-1]
        tokens = nn.Dense(features=self.config.hidden_size, use_bias=False, 
                          dtype=jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32,
                          param_dtype=jnp.float32)(tokens)
        
        ### DiT
        for _ in range(self.config.n_iterations):
            tokens = DiffusionTransformerBlock(self.config.dit_block, self.global_config) \
                (tokens, tokens_mask, tokens_rope_index.astype(jnp.int32), condition_emb)
            
        tokens = DiffusionTransformerOutput(**self.config.dit_output, 
                                            output_size=raw_emb_dim,
                                            global_config=self.global_config) \
            (tokens, condition_emb) ## (B, T, C)

        ### TODO: some other reshape / transformation maybe needed here
        return tokens