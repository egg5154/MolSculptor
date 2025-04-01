"""pre / post / residual
attention + transition + norm * 2
pre: q, m_q, optional(k, v, m_kv, z, m_z) ## control by config (self or cross attention)
    self or cross -> if self: causal or not
    self:
        (q(s_i) -> norm -> attention_embs -> optional(hak rope ipa) -> attention -> norm -> transition) * n -> norm
    cross:
        (q, k, v -> norm -> attention_embs -> attention -> norm -> transition) * n -> norm ## post process
post: 
    self:
        maybe norm -> (q -> attention_emb -> optional(hak rope ipa) -> attention -> norm -> transition -> norm) * n
    cross:
        like pre
residual: q, acc_q, ...
    ## q is updated by post norm
    q -> .... -> attention_out_q -> norm -> transition -> transition_out_q -> norm
    acc_q -> acc_q + attention_out_q + transition_out_q
    post process:
        w1 * q + w2 * norm(acc_q)
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from jax import Array
from ml_collections.config_dict import ConfigDict
from typing import Union, Optional, Tuple

from ..common.config import Config
from ..common.utils import gather_neighbor
from ..module.transformer import NormBlock, Transition
from ..module.attention import AttentionEmbedding, AttentionKernel, HyperAttentionEmbedding, PostAttention

class AttentionBlock(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, single_act, single_mask, rope_index = None):

        #### 1. Attention Embedding
        embedding_config = self.config.attention_embedding
        q, k, v, _ = AttentionEmbedding(embedding_config, self.global_config)(single_act)

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
        out_act = PostAttention(post_attention_config, self.global_config)(out_act, q)

        #### 5. dropout
        dropout_flag = self.global_config.dropout_flag
        out_act = nn.Dropout(
            rate=self.config.dropout_rate, deterministic=(not dropout_flag)
            )(out_act)

        return out_act

class TransitionBlock(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, act) -> Array:
        
        #### 1. Transition (GLU or FFN)
        act = Transition(self.config.transition, self.global_config)(act)

        #### 2. Dropout
        dropout_flag = self.global_config.dropout_flag
        act = nn.Dropout(
            rate=self.config.dropout_rate, deterministic=(not dropout_flag)
            )(act)

        return act

class PreNormTransformerBlock(nn.Module):

    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]
    hyper_lora_config: Union[Config, ConfigDict, None] = None

    def setup(self):
        
        ## basic config
        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.dropout_flag
        self.arr_dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
       
        ## setup modules
        _norm_method = self.global_config.norm_method # "layernorm" or "rmsnorm"
        _norm_small = self.global_config.norm_small   # epsilon
        self.transition_norm = NormBlock(_norm_method, _norm_small)
        self.attention_norm = NormBlock(_norm_method, _norm_small)

        ## attention embedding
        embedding_config = self.config.attention_embedding
        self.cross_attention_flag = True if embedding_config.attention_type == "cross" else False
        self.attention_embedding = AttentionEmbedding(embedding_config, self.global_config,
                                                      self.hyper_lora_config)

        ### hyper attention
        self.hyper_attention_flag = self.config.hyper_attention_flag
        if self.hyper_attention_flag:
            hyper_attention_config = self.config.hyper_attention_embedding
            self.hyper_attention_embedding = HyperAttentionEmbedding(hyper_attention_config, self.global_config)

        ### attention operation
        kernel_config = self.config.attention_kernel
        self.attention_kernel = AttentionKernel(kernel_config, self.global_config)
        post_attention_config = self.config.post_attention
        self.post_attention = PostAttention(post_attention_config, self.global_config, self.hyper_lora_config)

        ### transition
        transition_config = self.config.transition
        self.transition = Transition(transition_config, self.global_config, self.hyper_lora_config)

        ### dropout
        self.dropout_attention = nn.Dropout(rate=self.config.dropout_rate, deterministic=(not self.dropout_flag))
        self.dropout_transition = nn.Dropout(rate=self.config.dropout_rate, deterministic=(not self.dropout_flag))
        
        ### hyper lora 
        self.hyper_lora_flag = True if self.hyper_lora_config else False

    def __call__(self, 
                 s_i: Union[Array, Tuple],
                 m_i: Union[Array, Tuple],
                 m_j: Array = None,
                 z_ij: Array = None,
                 m_ij: Array = None,
                 n_i_or_r_i: Array = None,
                 hyper_var: Array = None):
        """transformer block for post/pre norm
        Inputs:
            s_i: single activation;
                for self attention: (B, Q, Fs)
                for cross attention: [(B, Q, Fs), (B, K, Fs)]
            m_i: mask for single activation;
                for self attention: (B, Q)
                for cross attention: [(B, Q), (B, K)]
            m_j: mask for neighbors, only needed in hyper-attention;
                for sparse mode: (B, Q, Qn)
                for dense mode: (B, Q, Q)
            z_ij: pair activation (B, Q, Qn, Fz);
            m_ij: mask for pair activation (B, Q, Qn);
            n_i_or_r_i: neighbor index (B, Q, Qn) or rope position index (B, Qn)            
        Outputs:
            single_act: (B, Q, Fs)
        NOTE: 
            z_ij, m_ij and n_i_or_r_i is only needed in hyper-attention.
        """
        
        hyper_var_ = {"hyper_var": hyper_var} if self.hyper_lora_flag else {}
        ## for residual connection
        ## if cross attention, then only use the first single activation
        out_i = s_i[0] if self.cross_attention_flag else s_i

        ## pre norm if needed
        s_i = jax.tree_map(self.attention_norm, s_i)

        ## attention embedding
        q_i, k_i, v_i, z_ij = self.attention_embedding(s_i, z_ij, **hyper_var_)

        ## hyper attention if needed
        ## if we use cross attention, then hyper attention embedding is not recommended
        if self.hyper_attention_flag:
            q_i, k_i = self.hyper_attention_embedding(q_i, k_i, m_j, z_ij, m_ij, n_i_or_r_i)

        ## attention kernel
        act = self.attention_kernel(q_i, k_i, v_i, None, m_i)
        act = self.post_attention(act, q_i, **hyper_var_)

        act = self.dropout_attention(act)
        out_i += act

        ## transition
        act = out_i
        act = self.transition_norm(act)
        act = self.transition(act, **hyper_var_)
        act = self.dropout_transition(act)
        out_i += act

        return out_i

class PostNormTransformerBlock(nn.Module):

    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]
    hyper_lora_config: Union[Config, ConfigDict, None] = None

    def setup(self):
        
        ## basic config
        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.dropout_flag
        self.arr_dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
       
        ## setup modules
        ## norm: both pre norm and post norm are supported
        _norm_method = self.global_config.norm_method # "layernorm" or "rmsnorm"
        _norm_small = self.global_config.norm_small   # epsilon
        self.transition_norm = NormBlock(_norm_method, _norm_small)
        self.attention_norm = NormBlock(_norm_method, _norm_small)

        ## attention embedding
        embedding_config = self.config.attention_embedding
        self.cross_attention_flag = True if embedding_config.attention_type == "cross" else False
        self.attention_embedding = AttentionEmbedding(embedding_config, self.global_config,
                                                      self.hyper_lora_config)

        ### hyper attention
        self.hyper_attention_flag = self.config.hyper_attention_flag
        if self.hyper_attention_flag:
            hyper_attention_config = self.config.hyper_attention_embedding
            self.hyper_attention_embedding = HyperAttentionEmbedding(hyper_attention_config, self.global_config)

        ### attention operation
        kernel_config = self.config.attention_kernel
        self.attention_kernel = AttentionKernel(kernel_config, self.global_config)
        post_attention_config = self.config.post_attention
        self.post_attention = PostAttention(post_attention_config, self.global_config, self.hyper_lora_config)

        ### transition
        transition_config = self.config.transition
        self.transition = Transition(transition_config, self.global_config, self.hyper_lora_config)

        ### dropout
        self.dropout_attention = nn.Dropout(rate=self.config.dropout_rate, deterministic=(not self.dropout_flag))
        self.dropout_transition = nn.Dropout(rate=self.config.dropout_rate, deterministic=(not self.dropout_flag))

        ## hyper lora flag
        self.hyper_lora_flag = True if self.hyper_lora_config else False

    def __call__(self, 
                 s_i: Union[Array, Tuple],
                 m_i: Union[Array, Tuple],
                 m_j: Array = None,
                 z_ij: Array = None,
                 m_ij: Array = None,
                 n_i_or_r_i: Array = None,
                 hyper_var: Array = None,):
        """transformer block for post/pre norm
        Inputs:
            s_i: single activation;
                for self attention: (B, Q, Fs)
                for cross attention: [(B, Q, Fs), (B, K, Fs)]
            m_i: mask for single activation;
                for self attention: (B, Q)
                for cross attention: [(B, Q), (B, K)]
            m_j: mask for neighbors, only needed in hyper-attention;
                for sparse mode: (B, Q, Qn)
                for dense mode: (B, Q, Q)
            z_ij: pair activation (B, Q, Qn, Fz);
            m_ij: mask for pair activation (B, Q, Qn);
            n_i_or_r_i: neighbor index (B, Q, Qn) or rope position index (B, Qn)            
        Outputs:
            single_act: (B, Q, Fs)
        NOTE: 
            z_ij, m_ij and n_i_or_r_i is only needed in hyper-attention.
        """

        hyper_var_ = {"hyper_var": hyper_var} if self.hyper_lora_flag else {}
        ## for residual connection
        ## if cross attention, then only use the first single activation
        out_i = s_i[0] if self.cross_attention_flag else s_i

        ## attention embedding
        q_i, k_i, v_i, z_ij = self.attention_embedding(s_i, z_ij, **hyper_var_)

        ## hyper attention if needed
        ## if we use cross attention, then hyper attention embedding is not recommended
        if self.hyper_attention_flag:
            q_i, k_i = self.hyper_attention_embedding(q_i, k_i, m_j, z_ij, m_ij, n_i_or_r_i)

        ## attention kernel
        act = self.attention_kernel(q_i, k_i, v_i, None, m_i)
        act = self.post_attention(act, q_i, **hyper_var_)
        act = self.dropout_attention(act)

        ## post norm
        out_i += act
        out_i = self.attention_norm(out_i)

        ## transition
        act = out_i
        act = self.transition(act, **hyper_var_)
        act = self.dropout_transition(act)

        ## post norm
        out_i += act
        out_i = self.transition_norm(out_i)

        return out_i

class ResiDualTransformerBlock(nn.Module):
    """ResiDual: from https://arxiv.org/abs/2304.14802
    """

    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]
    hyper_lora_config: Union[Config, ConfigDict, None] = None

    def setup(self):
        
        ## basic config
        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.dropout_flag
        self.arr_dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32
        
        ## setup modules
        ## norm
        _norm_method = self.global_config.norm_method # "layernorm" or "rmsnorm"
        _norm_small = self.global_config.norm_small   # epsilon
        self.transition_norm = NormBlock(_norm_method, _norm_small)
        self.attention_norm = NormBlock(_norm_method, _norm_small)

        ## attention embedding
        embedding_config = self.config.attention_embedding
        self.cross_attention_flag = True if embedding_config.attention_type == "cross" else False
        self.attention_embedding = AttentionEmbedding(embedding_config, self.global_config,
                                                      self.hyper_lora_config)

        ### hyper attention
        self.hyper_attention_flag = self.config.hyper_attention_flag
        if self.hyper_attention_flag:
            hyper_attention_config = self.config.hyper_attention_embedding
            self.hyper_attention_embedding = HyperAttentionEmbedding(hyper_attention_config, self.global_config)
        else:
            self.proj_z = nn.Dense(features = 1, dtype = self.arr_dtype)

        ### attention operation
        kernel_config = self.config.attention_kernel
        self.attention_kernel = AttentionKernel(kernel_config, self.global_config)
        post_attention_config = self.config.post_attention
        self.post_attention = PostAttention(post_attention_config, self.global_config, self.hyper_lora_config)

        ### transition
        transition_config = self.config.transition
        self.transition = Transition(transition_config, self.global_config, self.hyper_lora_config)

        ### dropout
        self.dropout_attention = nn.Dropout(rate=self.config.dropout_rate, deterministic=(not self.dropout_flag))
        self.dropout_transition = nn.Dropout(rate=self.config.dropout_rate, deterministic=(not self.dropout_flag))

        ## hyper lora
        self.hyper_lora_flag = True if self.hyper_lora_config else False

    def __call__(self, 
                 s_i: Union[Array, Tuple],
                 acc_s_i: Union[Array, Tuple],
                 m_i: Union[Array, Tuple],
                 m_j: Array = None,
                 z_ij: Array = None,
                 m_ij: Array = None,
                 n_i_or_r_i: Array = None,
                 hyper_var: Array = None,):
        """transformer block for post/pre norm
        Inputs:
            s_i: single activation;
                for self attention: (B, Q, Fs)
                for cross attention: [(B, Q, Fs), (B, K, Fs)]
            acc_s_i: accumulated single activation;
                for self attention: (B, Q, Fs)
                for cross attention: [(B, Q, Fs), (B, K, Fs)]
            m_i: mask for single activation;
                for self attention: (B, Q)
                for cross attention: [(B, Q), (B, K)]
            m_j: mask for neighbors, only needed in hyper-attention;
                for sparse mode: (B, Q, Qn)
                for dense mode: (B, Q)
            z_ij: pair activation (B, Q, Qn, Fz);
            m_ij: mask for pair activation (B, Q, Qn);
            n_i_or_r_i: neighbor index (B, Q, Qn) or rope position index (B, Qn)            
        Outputs:
            single_act: (B, Q, Fs)
        NOTE: 
            z_ij, m_ij and n_i_or_r_i is only needed in hyper-attention.
        """

        hyper_var_ = {"hyper_var": hyper_var} if self.hyper_lora_flag else {}
        ## for residual connection
        ## if cross attention, then only use the first single activation
        out_i = s_i[0] if self.cross_attention_flag else s_i

        ## attention embedding
        q_i, k_i, v_i, z_ij = self.attention_embedding(s_i, z_ij, **hyper_var_)

        ## hyper attention if needed
        ## if we use cross attention, then hyper attention embedding is not recommended
        if self.hyper_attention_flag:
            q_i, k_i = self.hyper_attention_embedding(q_i, k_i, m_j, z_ij, m_ij, n_i_or_r_i)
            b_ij = None
        else:
            if (z_ij is not None):
                b_ij = self.proj_z(z_ij).squeeze(-1) # (B, Q, Qn, H)
                b_ij = jnp.transpose(b_ij, (0, 3, 1, 2))
            else:
                b_ij = None

        ## attention kernel
        act = self.attention_kernel(q_i, k_i, v_i, b_ij, m_i)
        act = self.post_attention(act, q_i, **hyper_var_)
        act = self.dropout_attention(act)

        out_i += act
        acc_s_i += act
        out_i = self.transition_norm(out_i)

        ## transition
        act = out_i
        act = self.transition(act, **hyper_var_)
        act = self.dropout_transition(act)

        ## post norm if we need
        acc_s_i += act
        out_i += act
        out_i = self.attention_norm(out_i)

        return out_i, acc_s_i

