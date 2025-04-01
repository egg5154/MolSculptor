# Code for log-gaussian radial basis function.

import jax
import math
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from typing import Optional, Union, Tuple, List
from ml_collections import ConfigDict
from ..config import Config

class LogGaussianBasis(nn.Module):

    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]

    def setup(self):

        self.r_max = self.config.r_max
        self.r_min = self.config.r_min
        self.sigma = self.config.sigma
        self.delta = self.config.delta
        self.num_basis = self.config.num_basis
        self.rescale = self.config.rescale
        self.clip_distance = self.config.clip_distance
        self.r_ref = self.config.r_ref

        self.dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32

        if self.r_max <= self.r_min:
            raise ValueError("[utils/rbf/LogGaussianBasis] r_max should be larger than r_min.")
        
        self.r_range = self.r_max - self.r_min

        if self.num_basis is None and self.delta is None:
            raise TypeError('[utils/rbf/LogGaussianBasis] "num_basis" and "delta" cannot both be "None".')
        if self.num_basis is not None and self.num_basis <= 0:
            raise ValueError('[utils/rbf/LogGaussianBasis] "num_basis" must be larger than 0.')
        if self.delta is not None and self.delta <= 0:
            raise ValueError('[utils/rbf/LogGaussianBasis] "delta" must be larger than 0.')

        self.log_rmin = np.log(self.r_min/self.r_ref)
        self.log_rmax = np.log(self.r_max/self.r_ref)
        self.log_range = self.log_rmax - self.log_rmin
        if self.delta is None and self.num_basis is not None:
            self.offsets = jnp.linspace(self.log_rmin, self.log_rmax, self.num_basis)
        else:
            if self.num_basis is None:
                _num_basis = math.ceil(self.log_range / self.delta) + 1
                self.offsets = self.log_rmin + jnp.arange(0, _num_basis) * self.delta
            else:
                self.offsets = self.log_rmin + jnp.arange(0, self.num_basis) * self.delta
        
        self.coefficient = -0.5 * jnp.reciprocal(jnp.square(self.sigma))
        self.inv_ref = jnp.reciprocal(self.r_ref)

    def __call__(self, distance: jnp.ndarray) -> jnp.ndarray:
        """Compute gaussian type RBF.

        ## Args:
            distance (Array): Array of shape `(...)`. Data type is float.

        ## Returns:
            rbf (Array):      Array of shape `(..., K)`. Data type is float.

        """

        # cast
        distance, offsets, coefficient, inv_ref = \
            jax.tree_map(self.dtype, (distance, self.offsets, self.coefficient, self.inv_ref))

        if self.clip_distance:
            distance = jnp.clip(distance, self.r_min, self.r_max)
        
        # (...,) -> (..., 1)
        log_r = jnp.log(distance * self.inv_ref) ## Liyh: The main difference between jax and ms
        log_r = jnp.expand_dims(log_r, axis=-1)
        # (..., 1) - (..., K) -> (..., K)
        log_diff = log_r - self.offsets
        rbf = jnp.exp(self.coefficient * jnp.square(log_diff))

        if self.rescale:
            rbf = rbf * 2.0 - 1.0
        
        return rbf
    
    def __str__(self):
        return 'LogGaussianBasis<>'
