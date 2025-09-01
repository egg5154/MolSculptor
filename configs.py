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
hidden_size = 512
n_head = hidden_size // 32
n_iters = 12
num_basis = 256

qed_min, qed_max = (0.25, 0.95)
sa_min, sa_max = (2.0, 4.5)
logp_min, logp_max = (-2.0, 6.0)
mw_min, mw_max = (200.0, 500.0)

label_embedding_config = {
    'num_basis': num_basis,
    'hidden_size': hidden_size,
    'label_drop_flag': True,
    'label_drop_rate': 0.1,
    'clip_distance': False,
    'qed_min': qed_min, 'qed_max': qed_max, 'qed_sigma': (qed_max - qed_min) / num_basis,
    'sa_min': sa_min, 'sa_max': sa_max, 'sa_sigma': (sa_max - sa_min) / num_basis,
    'logp_min': logp_min, 'logp_max': logp_max, 'logp_sigma': (logp_max - logp_min) / num_basis,
    'mw_min': mw_min, 'mw_max': mw_max, 'mw_sigma': (mw_max - mw_min) / num_basis,
}

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
    'emb_label_flag': True,
    'hidden_size': hidden_size,
    'time_embedding': time_embedding_config,
    'label_embedding': label_embedding_config,
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
    'force_drop_rate': 1/4,
}

dit_config = ConfigDict(dit_config)
data_config = ConfigDict(data_config)
train_config = ConfigDict(train_config)
global_config = ConfigDict(global_config)