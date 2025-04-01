"""
A mlp module from haiku.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn

from ml_collections.config_dict import ConfigDict
from typing import Any, Callable, Iterable, Optional, Union, Tuple

from ..utils import get_activation

class MLP(nn.Module):
    """A multi-layer perceptron module."""

    output_sizes: Iterable[int]
    w_init: Optional[nn.initializers.Initializer] = nn.initializers.xavier_uniform()
    b_init: Optional[nn.initializers.Initializer] = nn.initializers.zeros_init()
    with_bias: bool = True
    activation: Union[Callable, str] = "relu"
    activate_final: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    dropout_rate: Optional[float] = None
    dropout_flag: bool = False

    @nn.compact
    def __call__(self,
                 inputs: jax.Array,
                 ) -> jax.Array:
        """Connects the module to some inputs.

        ## Args:
        inputs: A Tensor of shape ``[batch_size, input_size]``.

        ## Returns:
        The output of the model of size ``[batch_size, output_size]``.
        """

        layers = []
        output_sizes = tuple(self.output_sizes)
        for index, output_size in enumerate(output_sizes):
            layers.append(nn.Dense(features=output_size,
                                   kernel_init=self.w_init,
                                   bias_init=self.b_init,
                                   use_bias=self.with_bias,
                                   dtype=self.dtype,
                                   param_dtype=self.param_dtype,
                                   name=f"linear_{index}"))
        layers = tuple(layers)
        output_size = output_sizes[-1] if output_sizes else None

        num_layers = len(layers)
        out = inputs
        for i, layer in enumerate(layers):
            out = layer(out)
            if i < (num_layers - 1) or self.activate_final:
                if self.dropout_flag:
                    out = nn.Dropout(
                        rate = self.dropout_rate, deterministic = not self.dropout_flag)(out)
                out = get_activation(self.activation)(out)
        
        return out

class ResMLP(nn.Module):
    """A multi-layer perceptron module."""

    output_sizes: Iterable[int]
    w_init: Optional[nn.initializers.Initializer] = nn.initializers.xavier_uniform()
    b_init: Optional[nn.initializers.Initializer] = nn.initializers.zeros_init()
    with_bias: bool = True
    activation: Union[Callable, str] = "relu"
    activate_final: bool = False
    dtype: Any = jnp.float32
    param_dtype: Any = jnp.float32
    dropout_rate: Optional[float] = None

    @nn.compact
    def __call__(self,
                 inputs: jax.Array,
                 ) -> jax.Array:
        """Connects the module to some inputs.

        ## Args:
        inputs: A Tensor of shape ``[batch_size, input_size]``.

        ## Returns:
        The output of the model of size ``[batch_size, output_size]``.
        """

        layers = []
        output_sizes = tuple(self.output_sizes)
        for index, output_size in enumerate(output_sizes):
            layers.append(nn.Dense(features=output_size,
                                   kernel_init=self.w_init,
                                   bias_init=self.b_init,
                                   use_bias=self.with_bias,
                                   dtype=self.dtype,
                                   param_dtype=self.param_dtype,
                                   name=f"linear_{index}"))
        layers = tuple(layers)
        output_size = output_sizes[-1] if output_sizes else None

        num_layers = len(layers)
        out = nn.LayerNorm(
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            )(inputs)
        for i, layer in enumerate(layers):
            if i < (num_layers - 1):
                out += layer(out) ## res connection
                if self.dropout_rate is not None:
                    out = nn.Dropout(rate=self.dropout_rate)(out)
                out = get_activation(self.activation)(out)
            else: ## final layer
                out = layer(out)
                if self.activate_final:
                    out = get_activation(self.activation)(out)
        
        return out