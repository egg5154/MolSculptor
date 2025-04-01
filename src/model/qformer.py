"""Q-Former model"""

import os
import sys
import jax
import jax.numpy as jnp
import flax.linen as nn

sys.path.append(os.path.dirname(sys.path[0]))

from jax import Array
from ml_collections.config_dict import ConfigDict
from ..module.attention import AttentionEmbedding, HyperAttentionEmbedding, AttentionKernel, PostAttention
from ..module.transformer import NormBlock, Transition
from .transformer import AttentionBlock, TransitionBlock

class QFormerBlock(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, 
                 acc_act,
                 q_act, kv_act,
                 q_mask, kv_mask,
                 q_rope_index = None,)-> Array:
        
        #### set global config
        norm_method = self.global_config.norm_method
        norm_small = self.global_config.norm_small
        
        #### 1. Self Attention Block
        act, d_act, m_act = (q_act, q_act, q_mask)
        d_act = AttentionBlock(
            self.config.self_attention, self.global_config,
        )(d_act, m_act, q_rope_index)
        act += d_act
        acc_act += d_act
        act = NormBlock(norm_method, norm_small)(act)

        #### 2. Cross Attention Block
        act, d_act, m_act = act, (act, kv_act), (q_mask, kv_mask)
        d_act = AttentionBlock(
            self.config.cross_attention, self.global_config,
        )(d_act, m_act, None)
        act += d_act
        acc_act += d_act
        act = NormBlock(norm_method, norm_small)(act)

        #### 3. Feed Forward Block
        act, d_act = act, act
        d_act = TransitionBlock(
            self.config.transition_block, self.global_config,
        )(d_act)
        act += d_act
        acc_act += d_act
        act = NormBlock(norm_method, norm_small)(act)

        return act, acc_act
