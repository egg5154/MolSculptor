# Code for gaussian radial basis function.

import jax
import math
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from typing import Optional, Union, Tuple, List
from ml_collections import ConfigDict
from ..config import Config

class GaussianBasis(nn.Module):
    
    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]

    def setup(self):

        self.r_max = self.config.r_max
        self.r_min = self.config.r_min
        self.sigma = self.config.sigma
        self.delta = self.config.delta
        self.num_basis = self.config.num_basis
        self.clip_distance = self.config.clip_distance

        self.dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32

        assert self.r_max > self.r_min, "[utils/rbf/GaussianBasis] r_max should be larger than r_min."
        self.r_range = self.r_max - self.r_min

        if self.num_basis is None and self.delta is None:
            raise TypeError('[utils/rbf/GaussianBasis] "num_basis" and "delta" cannot both be "None".')
        if self.num_basis is not None and self.num_basis <= 0:
            raise ValueError('[utils/rbf/GaussianBasis] "num_basis" must be larger than 0.')
        if self.delta is not None and self.delta <= 0:
            raise ValueError('[utils/rbf/GaussianBasis] "delta" must be larger than 0.')
        
        self.coefficient = -0.5 * jnp.reciprocal(jnp.square(self.sigma))

        if self.delta is None and self.num_basis is not None:
            self.offsets = jnp.linspace(self.r_min, self.r_max, self.num_basis)
        else:
            if self.num_basis is None:
                _num_basis = math.ceil(self.r_range / self.delta) + 1
                self.offsets = self.r_min + jnp.arange(0, _num_basis) * self.delta
            else:
                self.offsets = self.r_min + jnp.arange(0, self.num_basis) * self.delta
    
    def __call__(self, distance: jnp.ndarray) -> jnp.ndarray:
        r"""Compute gaussian type RBF.

        ## Args: 
            distance (Array):                 Distance matrix. Shape: (A, A).
        ## Returns: 
            radial basis embedding (Array):   Embedding of distance matrix. Shape: (A, A, num_basis).

        """

        # cast
        distance, offsets, coeffient = jax.tree_map(self.dtype, (distance, self.offsets, self.coefficient))

        if self.clip_distance:
            distance = jnp.clip(distance, self.r_min, self.r_max)
        
        # (..., ) -> (..., 1)
        distance = jnp.expand_dims(distance, axis=-1)
        # (..., 1) - (..., num_basis) -> (..., num_basis)
        diff = distance - offsets
        # (..., num_basis) -> (..., num_basis)
        rbf = jnp.exp(coeffient * jnp.square(diff))

        return rbf

    def __str__(self) -> str:
        return 'GaussianBasis<>'
