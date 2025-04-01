"""Modules for transformer.
Contains:
    - NormBlock: basic block for layernorm/rmsnorm;
    - Transition: enhanced transition from AlphaFold2;
    - OuterProduct: outer product from AlphaFold2;
    - OuterDifference: outer difference from ESMFold;
    - NormBlock: basic block for layernorm/rmsnorm;
Marks:
    - B: batch size;
    - N: number of residues;
    - n: number of neighbors;
    - C: number of channels;
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from jax import Array
from typing import Union, Optional
from ml_collections import ConfigDict
from flax.linen.initializers import lecun_normal, zeros_init
from ..common.config import Config
from ..common.utils import get_activation, get_initializer, gather_neighbor
from ..common.layers.dense import HyperLoRADense

class NormBlock(nn.Module):

    norm_method: str = "layernorm"
    eps: float = 1e-6

    @nn.compact
    def __call__(self, x: Array):

        if self.norm_method == "layernorm":
            x_safe = jnp.float32(x)
            x_safe = nn.LayerNorm(epsilon=self.eps)(x_safe)
            return x_safe.astype(x.dtype)
        elif self.norm_method == "rmsnorm":
            x_safe = jnp.float32(x)
            x_safe = nn.RMSNorm(epsilon=self.eps)(x_safe)
            return x_safe.astype(x.dtype)
        else:
            raise ValueError(f"Unsupported norm method: {self.norm_method}")

class Transition(nn.Module):

    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]
    hyper_lora_config: Union[Config, ConfigDict, None] = None

    @nn.compact
    def __call__(self, z_ij: Array, hyper_var: Array = None):
        """Transition operation for pair activations.
        Input: 
            z_ij: shape of (B, N, n, C) or (N, n, C), pair activations;
        Output:
            z_ij: shape of (B, N, n, C) or (N, n, C), out pair activations;
        """

        ## ----- algorithm -----
        ## z_ij -> Norm(z_ij) -> FFN or GLU(z_ij)
        bf16_flag = self.global_config.bf16_flag
        arr_type = jnp.bfloat16 if bf16_flag else jnp.float32
        dropout_flag = self.global_config.dropout_flag

        act_fn = get_activation(self.config.act_fn)
        method = self.config.method
        assert method in ["ffn", "glu"], f"Unsupported method in transition: {method}."

        _dim_channel = z_ij.shape[-1]
        _dim_transition = self.config.transition_factor * _dim_channel
        kernel_initializer = get_initializer(self.config.kernel_initializer)
        ## for lora
        arg_dict = {'features': _dim_transition, 'use_bias': False, 'dtype': arr_type,
                    'param_dtype': jnp.float32, 'kernel_init': kernel_initializer()}
        
        hyper_lora_flag = True if self.hyper_lora_config else False
        if hyper_lora_flag:
            DenseModule = HyperLoRADense
            arg_dict = {**arg_dict, **self.hyper_lora_config.lora_dense_args, 'lora_dropout_flag': dropout_flag}
            hyper_var_ = {"hyper_var": hyper_var}
        else:
            DenseModule = nn.Dense
            hyper_var_ = {}
        if method == "ffn":
            # (..., N, n, C) -> (..., N, n, C*f) -> (..., N, n, C)
            z_ij = DenseModule(name="ffn1", **arg_dict)(z_ij, **hyper_var_)
            z_ij = act_fn(z_ij)
            arg_dict['features'] = _dim_channel
            z_ij = DenseModule(name="ffn2", **arg_dict)(z_ij, **hyper_var_)
        elif method == "glu":
            # (..., N, n, C) -> (..., N, n, C*f)
            h_ij = DenseModule(name="glu1", **arg_dict)(z_ij, **hyper_var_)
            # (..., N, n, C) -> (..., N, n, C*f)
            k_ij = DenseModule(name="glu2", **arg_dict)(z_ij, **hyper_var_)
            # (..., N, n, C*f) * (..., N, n, C*f) -> (..., N, n, C*f)
            z_ij = h_ij * act_fn(k_ij)
            # (..., N, n, C*f) -> (..., N, n, C)
            arg_dict['features'] = _dim_channel
            z_ij = DenseModule(name="glu_out", **arg_dict)(z_ij, **hyper_var_)

        return z_ij

class OuterProduct(nn.Module):

    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]

    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.dropout_flag
        self.norm_small = self.global_config.norm_small
        self.arr_dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        outerproduct_dim = self.config.outerproduct_dim
        output_dim = self.config.output_dim
        has_bias = self.config.has_bias ## default is True

        self.left_projection = nn.Dense(features=outerproduct_dim, kernel_init=lecun_normal(),
                                        use_bias=has_bias, bias_init=zeros_init(), dtype=self.arr_dtype)
        self.right_projection = nn.Dense(features=outerproduct_dim, kernel_init=lecun_normal(),
                                         use_bias=has_bias, bias_init=zeros_init(), dtype=self.arr_dtype)
        self.linear_output = nn.Dense(features=output_dim, kernel_init=zeros_init(),
                                      use_bias=has_bias, bias_init=zeros_init(), dtype=self.arr_dtype)
    
    def __call__(self, s_i: Array, s_j: Array, m_i: Array, m_j: Array):
        """Outer product operation from AlphaFold2.
        Inputs:
            s_i: shape of (B, N, C) or (N, C), single act;
            s_j: shape of (B, N, n, C) or (N, n, C), single act for neighbors;
            m_i: shape of (B, N) or (N,), single mask;
            m_j: shape of (B, N, n) or (N, n), single mask for neighbors;
        outputs:
            act: shape of (B, N, n, C2) or (N, n, C2), pair update;
        """

        act = s_i
        _bs, _nres, _c = s_i.shape
        _n = s_j.shape[-2]

        # (B, N, C) -> (B, N, C1)
        left_act = m_i[..., None] * self.left_projection(act)
        _c1 = left_act.shape[-1]
        # (B, N, n, C) -> (B, N, n, C1)
        right_act = m_j[..., None] * self.right_projection(s_j)
        # (B, N, C1) -> (B, C1, N) -> (B, C1*N, 1)
        left_act = jnp.swapaxes(left_act, -1, -2).reshape(-1, _c1*_nres, 1)
        # (B, N, n, C1) -> (B, N, n*C1) -> (B, C1, N, n*C1) -> (B, C1*N, n*C1)
        right_act = jnp.reshape(right_act, (_bs, _nres, -1))
        right_act = right_act[:, None, :, :].repeat(_c1, -3)
        right_act = jnp.reshape(right_act, (_bs, _c1*_nres, _n*_c1))
        # (B, C1*N, 1) * (B, C1*N, n*C1) -> (B, C1*N, n*C1) -> (B, C1, N, n, C1)
        act = left_act * right_act
        act = jnp.reshape(act, (_bs, _c1, _nres, _n, _c1))
        # (B, C1, N, n, C1) -> (B, N, n, C1, C1) -> (B, N, n, C1*C1)
        act = jnp.transpose(act, (0, 2, 3, 1, 4))
        act = jnp.reshape(act, (_bs, _nres, _n, _c1*_c1))
        # (B, N, n, C1*C1) -> (B, N, n, C2), C2 = output_dim
        act = self.linear_output(act)

        return act

class OuterDifference(nn.Module):

    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]

    def setup(self):

        self.bf16_flag = self.global_config.bf16_flag
        self.dropout_flag = self.global_config.dropout_flag
        self.norm_small = self.global_config.norm_small
        self.arr_dtype = jnp.bfloat16 if self.bf16_flag else jnp.float32

        outerdiff_dim = self.config.outerdiff_dim
        output_dim = self.config.output_dim
        has_bias = self.config.has_bias ## default is True

        self.left_projection = nn.Dense(features=outerdiff_dim, kernel_init=lecun_normal(),
                                        use_bias=has_bias, bias_init=zeros_init(), dtype=self.arr_dtype)
        self.right_projection = nn.Dense(features=outerdiff_dim, kernel_init=lecun_normal(),
                                         use_bias=has_bias, bias_init=zeros_init(), dtype=self.arr_dtype)
        self.linear_output = nn.Dense(features=output_dim, kernel_init=zeros_init(),
                                      use_bias=has_bias, bias_init=zeros_init(), dtype=self.arr_dtype)
    
    def __call__(self, s_i: Array, s_j: Array, m_i: Array, m_j: Array):
        """Outer difference operation from ESMFold.
        Inputs:
            s_i: shape of (B, N, C) or (N, C), single act;
            s_ij: shape of (B, N, n, C) or (N, n, C), single act for neighbors;
            m_i: shape of (B, N) or (N,), single mask;
        outputs:
            act: shape of (B, N, n, C2) or (N, n, C2), pair update;
        """

        act = s_i
        _bs, _nres, _c = s_i.shape
        _n = s_j.shape[-2]

        # (B, N, C) -> (B, N, C1) C1 = outerdiff_dim
        left_act = m_i[..., None] * self.left_projection(act)
        # (B, N, n, C) -> (B, N, n, C1)
        right_act = m_j[..., None] * self.right_projection(s_j)
        # (B, N, C1) -> (B, N, 1, C1)
        left_act = jnp.expand_dims(left_act, axis=-2)
        # (B, N, 1, C1) - (B, N, n, C1) -> (B, N, n, C1) -> (B, N, n, C2)
        act = left_act - right_act
        act = self.linear_output(act) # C2 = output_dim

        return act