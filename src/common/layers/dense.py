# Hyper Dense Module

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Union, Callable, Optional, Any
from flax.linen.initializers import lecun_normal, zeros_init, truncated_normal
from ..utils import get_activation
import math

Dtype = Any

def _l2_normalize(x, axis=-1, epsilon=1e-12):
    return x / jnp.sqrt(
        jnp.maximum(jnp.sum(x**2, axis=axis, keepdims=True), epsilon))
    
def safe_l2_normalize(x, axis=-1, epsilon=1e-12):
    _dtype = x.dtype
    x = x.astype(jnp.float32)
    x = _l2_normalize(x, axis=axis, epsilon=epsilon)   
    return x.astype(_dtype)

class DensewithAct(nn.Module):
    r"""A dense layer with activation function.
    
    ## Args:
        dim_out (int): Dimension of output vectors.

        activation (str, Callable): Activation function. Default: 'relu'.
    """

    dim_out: int
    use_bias: bool = True
    activation: Union[Callable, str] = "relu"
    d_type: Optional[Dtype] = jnp.float32
    kernel_init: Callable = lecun_normal()
    bias_init: Callable = zeros_init()

    @nn.compact
    def __call__(self, x):
        act_fn = get_activation(self.activation)
        linear_fn = nn.Dense(self.dim_out, self.use_bias, self.d_type, kernel_init=self.kernel_init, bias_init=self.bias_init)

        return act_fn(linear_fn(x)) 


class HyperLoRADense(nn.Module):
    r"""A dense layer with LoRA modulation.
    ### Formula:
        out = W @ x + dW @ x = W @ x + dW_left @ dW_right @ x
    """

    features: int
    use_bias: bool = True
    activation: Union[Callable, str] = lambda x: x
    dtype: Optional[Dtype] = jnp.float32
    param_dtype: Optional[Dtype] = jnp.float32
    kernel_init: Callable = lecun_normal()
    bias_init: Callable = zeros_init()
    lora_rank: int = 4
    lora_alpha: Union[int, None] = None
    lora_dropout_rate: float = 0.0
    lora_dropout_flag: bool = False 
    eps: float = 1e-6
    norm_method: str = "empirical" # "l2"                          
    
    @nn.compact
    def __call__(self, x, hyper_var):

        ## debug
        # print("hyper var shape", hyper_var.shape)

        x_baseline = nn.Dense(
            self.features, 
            self.use_bias,
            dtype = self.dtype,
            param_dtype = self.param_dtype,
            kernel_init = self.kernel_init, 
            bias_init = self.bias_init
        )(x)
        
        lora_layer_a = nn.Dense(x.shape[-1] * self.lora_rank, 
                                dtype = self.dtype, param_dtype = self.param_dtype,
                                use_bias = False, kernel_init = lecun_normal())
        lora_layer_b = nn.Dense(self.features * self.lora_rank, 
                                dtype = self.dtype, param_dtype = self.param_dtype,
                                use_bias = False, kernel_init = zeros_init()) # zeros_init())
        
        #### get lora vectors        
        lora_a = jnp.reshape(lora_layer_a(hyper_var), hyper_var.shape[:-1] + (self.lora_rank, x.shape[-1])) 
        lora_b = jnp.reshape(lora_layer_b(hyper_var), hyper_var.shape[:-1] + (self.lora_rank, self.features))

        ##### l2 norm, control the Frobenius norm of the matrix 
        if self.norm_method == "l2":
            lora_a = safe_l2_normalize(lora_a, axis=-1, epsilon=self.eps)
            lora_b = safe_l2_normalize(lora_b, axis=-1, epsilon=self.eps)
        else:
            ### emperical norm 
            empirical_norm_factor_in = math.sqrt(1.0 / (x.shape[-1] * self.lora_rank))
            empirical_norm_factor_out = math.sqrt(1.0 / (self.features * self.lora_rank))
            lora_a = lora_a * empirical_norm_factor_in
            lora_b = lora_b * empirical_norm_factor_out
        
        x = jnp.einsum("b...ij, bkj -> b...ik", nn.Dropout(rate = self.lora_dropout_rate, 
                                                           deterministic = not self.lora_dropout_flag)(x), lora_a) # (B, ?, rank)
        x = jnp.einsum("b...ij, bjk -> b...ik", x, lora_b)
        
        if self.lora_alpha:
            lora_scaling = self.lora_alpha / self.lora_rank
        else:
            lora_scaling = 1.0
        x = x_baseline + lora_scaling * x
        return get_activation(self.activation)(x)
