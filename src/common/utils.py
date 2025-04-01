"""Common utilities.
Contains:
    - get_activation: get activation function by name;
    - ShiftedSoftplus: shifted softplus activation;
    - ssp: shifted softplus activation;
    - get_initializer: get initializer function by name;
    - gather_neighbor: get neighbor features from input based on neighbor index;
"""

import jax
import jax.numpy as jnp
import numpy as np

from flax import linen as nn
from typing import Any, Optional, Union, Callable
from .rbf.gaussian import GaussianBasis
from .rbf.bessel import BesselBasis, NormBesselBasis
from .rbf.loggaussian import LogGaussianBasis

## Shifted Softplus
class ShiftedSoftplus(nn.Module):

    epsilon: float = 1e-6

    @nn.compact
    def __call__(self, x):
        return nn.softplus(x + self.epsilon) - jnp.log(2.0)

@jax.jit
def ssp(x, epsilon: float = 1e-6):
    return nn.softplus(x + epsilon) - jnp.log(2.0)

_activation_dict = {
    'relu': nn.relu,
    'relu6': nn.relu6,
    'sigmoid': nn.sigmoid,
    'softplus': nn.softplus,
    'silu': nn.silu,
    'swish': nn.swish,
    'leaky_relu': nn.leaky_relu,
    'gelu': nn.gelu,
    'ssp': ssp,
}

def get_activation(name):
    """get activation function by name"""
    if name is None:
        raise ValueError("Activation name cannot be None!")
    if isinstance(name, str):
        if name.lower() in _activation_dict.keys():
            return _activation_dict[name.lower()]
        raise ValueError(
            "The activation corresponding to '{}' was not found.".format(name))
    if isinstance(name, Callable):
        return name
    raise TypeError("Unsupported activation type '{}'.".format(type(name)))

_init_dict = {
    'lecun_normal': nn.initializers.lecun_normal,
    'lecun_uniform': nn.initializers.lecun_uniform,
    'glorot_normal': nn.initializers.glorot_normal,
    'glorot_uniform': nn.initializers.glorot_uniform,
    'he_normal': nn.initializers.he_normal,
    'he_uniform': nn.initializers.he_uniform,
    'kaiming_normal': nn.initializers.kaiming_normal,
    'kaiming_uniform': nn.initializers.kaiming_uniform,
    'zeros': nn.initializers.zeros_init,
    'ones': nn.initializers.ones_init,
    'constant': nn.initializers.constant,
    'normal': nn.initializers.normal,
    'uniform': nn.initializers.uniform,
    'xavier_uniform': nn.initializers.xavier_uniform,
    'xavier_normal': nn.initializers.xavier_normal,
}

def get_initializer(name):
    """get initializer function by name"""
    if name is None:
        raise ValueError("Initializer name cannot be None!")
    if isinstance(name, str):
        if name.lower() in _init_dict.keys():
            return _init_dict[name.lower()]
        raise ValueError(
            "The initializer corresponding to '{}' was not found.".format(name))
    if isinstance(name, Callable):
        return name
    raise TypeError("Unsupported initializer type '{}'.".format(type(name)))

_rbf_dict = {
    'gaussian': GaussianBasis,
    'bessel': BesselBasis,
    'norm_bessel': NormBesselBasis,
    'log_gaussian': LogGaussianBasis,
}

def get_rbf(name):
    """get rbf function by name"""
    if name is None:
        raise ValueError("RBF name cannot be None!")
    if isinstance(name, str):
        if name.lower() in _rbf_dict.keys():
            return _rbf_dict[name.lower()]
        raise ValueError(
            "The RBF corresponding to '{}' was not found.".format(name))
    if isinstance(name, Callable):
        return name
    raise TypeError("Unsupported RBF type '{}'.".format(type(name)))

def gather_neighbor(input: jnp.ndarray, neighbor_index: jnp.ndarray, is_pair: bool = True):
    """Get neighbor features from input based on neighbor index.
    Args:
        input: jnp.ndarray, (R, N, F) or (R, N, N, F)
        neighbor_index: jnp.ndarray, (R, N, n)
    Returns:
        out: jnp.ndarray, (R, N, n, F)   
    """
    
    if not is_pair:
        # (R, N, F) -> (R, 1, N, F) -> (R, N, N, F)
        n_res = input.shape[-2]
        input = jnp.expand_dims(input, axis=-3)
        input = jnp.repeat(input, n_res, axis=-3)
    # (R, N, N, F) -> (R, N, n, F)
    batch_size, n_res, n_res, c = input.shape
    input = jnp.reshape(input, (-1, n_res, c))
    neighbor_index = jnp.reshape(neighbor_index, (batch_size*n_res, -1))
    out = jax.vmap(jnp.take, in_axes=(0, 0, None))(input, neighbor_index, 0)
    out = jnp.reshape(out, (batch_size, n_res, -1, c))
    return out

def safe_l2_normalize(x, axis = -1, epsilon = 1e-8, safe_dtype = jnp.float32):
    """
    Performs L2 normalization with an epsilon for numerical stability.
    
    Args:
        x: Input array to normalize.
        axis: Axis along which to normalize. Default is -1 (last axis).
        epsilon: Small constant to add for numerical stability. Default is 1e-8.
        
    Returns:
        L2-normalized array with the same shape as `x`.
    """
    dtype = x.dtype
    x = x.astype(safe_dtype)
    l2_norm = jnp.sqrt(jnp.sum(jnp.square(x), axis=axis, keepdims=True) + epsilon)
    return (x / l2_norm).astype(dtype)

