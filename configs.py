from ml_collections.config_dict import ConfigDict

global_config = {
    "sparse_flag": False,
    "bf16_flag": True,
    "dropout_flag": True,
    "norm_small": 1e-6,
    "norm_method": "layernorm",
    "remat_flag": False,
    "test_flag": False,
}

train_config = {
    "diffusion_timesteps": 500,
    'learning_rate': {
        'min': 1e-5,
        'max': 2e-4,
        'warmup_steps': 10000,
        'decay_steps': 500000,
    },
    'weight_decay': 1e-4,
}

###### set hyperparameters here ######
hidden_size = 256
n_head = hidden_size // 32
n_iters = 6

time_embedding_config = {
    'hidden_size': hidden_size, 
    'frequency_embedding_size': hidden_size,
}

attention_config = {
    "attention_embedding": {
        "attention_type": "self",
        "dim_feature": hidden_size,
        "n_head": n_head,
        "embedding_pair_flag": False,
        "kernel_initializer": "glorot_uniform",
    },

    "hyper_attention_flag": True,
    "hyper_attention_embedding": {
        "kernel_type": "rope",
        "split_rope_flag": True,
    },
    
    "attention_kernel": {
        "attention_type": "self",
        "flash_attention_flag": False,
        "has_bias": False,
        "causal_flag": False,
        "block_q": 64,
        "block_k": 64,
    },

    "post_attention": {
        "out_dim": hidden_size,
        "gating_flag": False,
    },
    "dropout_rate": 0.01,
}

transition_config = {
    'transition': {
        "method": "glu",
        "transition_factor": 4,
        "kernel_initializer": "xavier_uniform",
        "act_fn": "gelu",
    },

    'dropout_rate': 0.01,
}

adaLN_config = {
    'hidden_size': hidden_size,
    'activation': 'silu',
}

dit_config = {
    'n_iterations': n_iters,
    'emb_label_flag': False,
    'hidden_size': hidden_size,
    'time_embedding': time_embedding_config,
    'dit_block': 
        {
            'attention': attention_config,
            'transition': transition_config,
            'adaLN': adaLN_config,
        },
    'dit_output': 
        {
            'hidden_size': hidden_size, 
        }
}

data_config = {
    'n_query_tokens': 16,
    'latent_dim': 32,
}

dit_config = ConfigDict(dit_config)
data_config = ConfigDict(data_config)
train_config = ConfigDict(train_config)
global_config = ConfigDict(global_config)