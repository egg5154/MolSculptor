import os
import sys

sys.path.append(os.path.dirname(sys.path[0]))

import jax
import jax.numpy as jnp
import flax.linen as nn

from jax import Array
from ml_collections.config_dict import ConfigDict
from typing import Union, Optional, Tuple

def safe_norm(x, axis = -1):
    return x / (jnp.sum(jnp.square(x), axis = axis, keepdims = True) + 1e-6)

class Quantizer(nn.Module):

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, f,):

        ### inputs: f: [B, N, C]
        B, N, C = f.shape
        n_patches = self.config.n_patches
        codebook = jnp.asarray(
            self.variables['params']['codebook'] ### Toodoo: why this?
        )
        for si, pn in enumerate(n_patches - 1):

            if self.config.using_znorm:
                ## (B, N, C) -> (B, pn, C) -> (B*pn, C)
                f_res = jax.image.resize(f, (B, pn, C), method='nearest')
                f_res = safe_norm(f_res, axis = -1).reshape(-1, C)
                ## (B*pn, C) @ (C, )
                idx_n = jnp.argmax(
                    jnp.dot(f_res, safe_norm(codebook, axis = -1).T), axis = -1,
                )
            else:
                raise NotImplementedError
                