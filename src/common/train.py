"""Tools for training.
"""

import jax
import optax
import jax.numpy as jnp
import numpy as np

from flax.training import common_utils
from typing import Union, Tuple, List, Optional
from jax import nn as nn
from jax.numpy import ndarray as ndarray
from optax import Schedule

## Learning schedule
def polynomial_decay_schedule(init_value: float,
                              power: float,
                              transition_begin: int = 0,) -> Schedule:
    
    def schedule(count):
        count += transition_begin
        return init_value * jnp.power(count, power)

    return schedule
    
def transformer_schedule(learning_rate: float = 1.0,
                         warmup_steps: int = 4000,
                         dimension: int = 1,):
    
    dim_scale = np.power(dimension, -0.5)
    max_lr = learning_rate * dim_scale * np.power(warmup_steps, -0.5)
    print(f"Using transformer learning rate, max rate is: {max_lr:.4e}")

    warmup_fn = optax.linear_schedule(init_value=0.0,
                                      end_value=max_lr,
                                      transition_steps=warmup_steps)
    decay_fn = polynomial_decay_schedule(init_value=learning_rate * dim_scale,
                                         power=-0.5,
                                         transition_begin=warmup_steps,)
    schedule_fn = optax.join_schedules(schedules=[warmup_fn, decay_fn],
                                       boundaries=[warmup_steps])
    
    return schedule_fn