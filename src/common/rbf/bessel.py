# Code for bessel radial basis function.

import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn

from typing import Optional, Union
from flax.linen.initializers import constant
from ml_collections import ConfigDict
from ..config import Config

class BesselBasis(nn.Module):
    
    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]
    name: str = "bessel_basis"

    @nn.compact
    def __call__(self, distance: jax.Array) -> jax.Array:

        r_max = self.config.r_max
        num_basis = self.config.num_basis
        trainable = self.config.trainable

        dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        param_dtype = jnp.float32

        assert r_max != 0, "[utils/rbf/BesselBasis] r_max should not be 0!"
        prefactor = 2.0 / r_max

        bessel_weights = jnp.linspace(1.0, num_basis, num_basis) * jnp.pi
        bessel_weights = self.param("bessel_weights", constant(bessel_weights), (num_basis,), param_dtype)
        if not trainable:   
            bessel_weights = jax.lax.stop_gradient(bessel_weights)
        
        # cast dtypes
        distance, bessel_weights = jax.tree_map(dtype, (distance, bessel_weights))
        # (..., ) -> (..., 1)
        distance = jnp.expand_dims(distance, axis=-1)
        # (..., 1) -> (..., num_basis)
        bessel_distance = bessel_weights * distance
        bessel_distance = bessel_distance / r_max
        numerator = jnp.sin(bessel_distance)
        ret = prefactor * (numerator / distance)

        return ret
    
class NormBesselBasis(BesselBasis):

    config: Union[Config, ConfigDict]
    global_config: Union[Config, ConfigDict]
    name: str = "norm_bessel_basis"

    @nn.compact
    def __call__(self, distance: jax.Array) -> jax.Array:

        r_max = self.config.r_max
        norm_num = self.config.norm_num

        dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32

        rs = jnp.linspace(0, r_max, norm_num + 1, dtype=dtype)
        bs = super().__call__(rs)
        basis_mean = jnp.mean(bs, axis=0)
        basis_std = jnp.std(bs, axis=0)

        basis_dis = super().__call__(distance)
        ret = (basis_dis - basis_mean) / basis_std

        return ret