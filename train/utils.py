import jax 
import jax.numpy as jnp
import numpy as np
import pickle as pkl
import concurrent.futures

def parameter_weight_decay(params):
    """Apply weight decay to parameters."""
    
    loss = jax.tree_util.tree_map(
        lambda p: jnp.mean(
                jnp.square(p.reshape(-1))
            ) if p.ndim == 2 else 0, params)
    loss = jnp.sum(
        jnp.array(jax.tree_util.tree_leaves(loss))
    )
    
    return loss

def split_multiple_rng_keys(rng_key, num_keys):
    rng_keys = jax.random.split(rng_key, num_keys + 1)
    return rng_keys[:-1], rng_keys[-1]

def print_nested_dict(d, prefix=""):
    for k, v in d.items():
        if isinstance(v, dict):
            print("{}{}:".format(prefix, k))
            print_nested_dict(v, prefix=prefix+"\t")
        else:
            print("{}{}: {}".format(prefix, k, v))

def print_net_params(params: dict):
    """Print all params with shape like a tree."""
    count = 0
    def _print_net_params(params: dict, ret: int = 0,):
        for k, v in params.items():
            if isinstance(v, dict):
                print(" "*ret + f"{k}:")
                _print_net_params(v, ret+2)
            else:
                print(" " * (ret+2) + f"{k}: {v.shape}")
    _print_net_params(params)
    param_arrays = jax.tree_util.tree_leaves(params)
    for p in param_arrays:
        count += p.size
    print(f"Total number of parameters: {count}")

def print_net_params_count(params: dict, exclude_paths = []):
    count = 0
    param_arrays = jax.tree_util.tree_leaves_with_path(params)
    for p in param_arrays:
        path, arr = p
        _size = arr.size
        for exclude_path in exclude_paths:
            if jax.tree_util.DictKey(exclude_path) in path:
                _size = 0
        count += _size
    # print(f"Total number of parameters: {count}, with excluded paths: {exclude_paths}.")
    return count

@jax.jit
def psum_tree(values):
    return jax.tree_util.tree_map(
        lambda x: jax.lax.psum(x, axis_name="i"), values
    )

@jax.jit
def pmean_tree(values):
    return jax.tree_util.tree_map(
        lambda x: jax.lax.pmean(x, axis_name="i"), values
    )

def gamma_schdule(init_value, peak_value, warmup_steps, total_steps):

    ### cosine raise schedule
    warmup_steps = min(warmup_steps, total_steps)
    warmup_gamma = init_value + (peak_value - init_value) * np.cos(
        (np.arange(warmup_steps) / warmup_steps - 1) * np.pi * 0.5 
        )
    
    res_steps = total_steps - warmup_steps
    res_gamma = np.ones(res_steps) * peak_value

    gamma = np.concatenate([warmup_gamma, res_gamma]) ## (T,)

    return gamma
    
def weight_schedule():

    raise NotImplementedError

def cosine_warmup_schedule(step_it, min_val, max_val, warmup_steps, decay_steps):

    #### linear
    linear_val = min_val + (max_val - min_val) * step_it / warmup_steps
    cosine_val = min_val + 0.5 * (max_val - min_val) * (1 + jnp.cos(jnp.pi * (step_it - warmup_steps) / decay_steps))
    cosine_val = jnp.maximum(cosine_val, min_val)

    return jnp.minimum(cosine_val, linear_val)

def cosine_const_schedule(step_it, min_val, max_val, const_steps, decay_steps):

    #### linear
    cosine_val = min_val + 0.5 * (max_val - min_val) * (1 + jnp.cos(jnp.pi * (step_it - const_steps) / decay_steps))
    cosine_val = jnp.maximum(cosine_val, min_val)

    return jnp.where(step_it < const_steps, max_val, cosine_val)

def exp_warmup_const_schedule(step_it, min_val, max_val, warmup_steps):

    a = min_val
    b = jnp.log(max_val / min_val) / warmup_steps
    return jnp.minimum(a * jnp.exp(b * step_it), max_val)