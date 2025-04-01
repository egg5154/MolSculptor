"""Example configuration file for transformer blocks.
"""

import jax
import jax.numpy as jnp

from ml_collections.config_dict import ConfigDict

global_config = {
    "sparse_flag": True,
    "bf16_flag": False,
    "dropout_flag": False,
    "norm_small": 1e-6,
    "norm_method": "layernorm",
}

transformer_config = {
    "attention_embedding": {
        "attention_type": "self",
        "dim_feature": 128,
        "n_head": 4,
        "embedding_pair_flag": True,
    },
    
    ## hyper attention
    "hyper_attention_flag": True,
    "hyper_attention_embedding": {
        "dim_r": 2,
        "kernel_type": "hak",
    },

    ## attention kernel
    "attention_kernel": {
        "flash_attention_flag": False,
        "has_bias": False,
        "causal_flag": False,
    },

    ## post attention
    "post_attention": {
        "out_dim": 128,
        "gating_flag": True,
    },

    ## transition
    "transition": {
        "method": "ffn",
        "transition_factor": 4,
        "kernel_initializer": "xavier_uniform",
        "act_fn": "relu",
    },
}

transformer_config = ConfigDict(transformer_config)
global_config = ConfigDict(global_config)