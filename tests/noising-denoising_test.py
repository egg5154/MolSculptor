"""
    In this script, we test the retain propotion of the denoising process.
"""

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))

import jax
import jax.numpy as jnp
import numpy as np
import pickle as pkl
import jax.tree_util as jtu

from tqdm import tqdm
from ml_collections import ConfigDict
from train.scheduler import GaussianDiffusion
from src.model.diffusion_transformer import DiffusionTransformer
from train.inference import InferEncoder, InferDecoder, Inferencer, \
    tokens2smiles, smi2graph_features, standardize
from src.common.utils import safe_l2_normalize
from utils import expand_batch_dim

def main(args):

    with open(args.init_molecule_path, 'rb') as f:
        test_molecules = pkl.load(f)
        test_molecules = expand_batch_dim(128, test_molecules)
    print(jtu.tree_map(lambda x: x.shape, test_molecules))
    dbs = test_molecules['smiles'].shape[0]

    ## load config
    with open(args.config_path, 'rb') as f:
            config_dicts = pkl.load(f)
    global_config = ConfigDict(config_dicts['global_config'])
    net_config = ConfigDict(config_dicts['net_config'])
    train_config = ConfigDict(config_dicts['train_config'])
    global_config.dropout_flag = False
    DIFFUSION_TIMESTEPS = train_config.diffusion_timesteps
    # vae config
    with open(args.vae_config_path, 'rb') as f:
        config_dicts = pkl.load(f)
    vae_config = ConfigDict(config_dicts['net_config'])
    data_config = ConfigDict(config_dicts['data_config'])
    vae_global_config = ConfigDict(config_dicts['global_config'])
    vae_global_config.dropout_flag = False
    
    #### load net
    dit_net = DiffusionTransformer(net_config, global_config)
    scheduler = GaussianDiffusion(train_config,)
    encoding_net = InferEncoder(vae_config, vae_global_config)
    decoding_net = InferDecoder(vae_config, vae_global_config)

    #### load params
    with open(args.vae_params_path, 'rb') as f:
        vae_params = pkl.load(f)
        vae_params = jtu.tree_map(lambda x: jnp.asarray(x), vae_params)
    encoder_params = {
        'Encoder_0': vae_params['params']['generator']['Encoder_0'],
        'Dense_0': vae_params['params']['generator']['Dense_0'],
    }
    decoder_params = {
        'Decoder_0': vae_params['params']['generator']['Decoder_0'],
    }
    with open(args.params_path, 'rb') as f:
        params = pkl.load(f)
        params = jtu.tree_map(jnp.asarray, params)

    #### set inferencer
    beam_size = args.beam_size
    npt = data_config['n_query_tokens']
    default_infer_config = {
        'sampling_method': args.sampling_method,
        'device_batch_size': dbs,
        'n_seq_length': data_config['n_pad_token'],
        'beam_size': beam_size,
        'bos_token': data_config['bos_token_id'],
        'eos_token': data_config['eos_token_id'],
        'n_local_device': jax.local_device_count(),
        'num_prefix_tokens': data_config['n_query_tokens'],
        'step_limit': 160,
    }
    infer_config = ConfigDict(default_infer_config)
    inferencer = Inferencer(
        encoding_net, decoding_net, encoder_params, decoder_params, infer_config)

    #### define encoder & decoder functions
    def encoder_function(graph_features):
        return inferencer.jit_encoding_graphs(graph_features)

    # load alphabet
    with open(args.alphabet_path, 'rb') as f:
        alphabet: dict = pkl.load(f)
        alphabet = alphabet['symbol_to_idx']
    reverse_alphabet = {v: k for k, v in alphabet.items()}
    def decoder_function(latent_tokens,):
        ## latent_tokens: (dbs, npt, d), cached_smiles: [(dbs,), ...]
        dbs, npt, dim = latent_tokens.shape
        latent_tokens = safe_l2_normalize(
            latent_tokens / np.sqrt(dim), axis = -1) ## reverse scale
        ## (dbs*bm, n_seq)
        output_tokens, aux = inferencer.beam_search(
            step = 0, cond = latent_tokens,)
        output_tokens = np.asarray(output_tokens, np.int32)
        ## (dbs*bm,) -> (dbs, bm)
        output_smiles = [
            tokens2smiles(t, reverse_alphabet) for t in output_tokens]
        output_smiles = np.asarray(
            output_smiles, object).reshape(dbs, beam_size)
        ## check if valid
        sanitized_output_smiles = np.empty((dbs, beam_size), object)
        for i_ in range(dbs):
            ## search for beam_size, from the most probable one
            ## if one is valid, then break.
            for j_ in range(beam_size - 1, -1, -1):
                smi_ = output_smiles[i_, j_]
                if smi_:
                    sanitized_output_smiles[i_, j_] = smi_
                else:
                    sanitized_output_smiles[i_, j_] = 'CC' ## default
                    # break
        return {'smiles': sanitized_output_smiles,}
    
    def noise(x, time, rng_key):
        """q(x_t | x_0)"""
        time = jnp.full((x.shape[0],), time) ## (dbs,)
        rng_key, sub_key = jax.random.split(rng_key)
        x = scheduler.q_sample(x, time, jax.random.normal(sub_key, x.shape))
        return x, rng_key
    jit_noise = jax.jit(noise)

    def noise_step(x, time, rng_key):
        time = jnp.full((x.shape[0],), time) ## (dbs,)
        rng_key, sub_key = jax.random.split(rng_key)
        x = scheduler.q_sample_step(x, time, jax.random.normal(sub_key, x.shape))
        return x, rng_key
    jit_noise_step = jax.jit(noise_step)

    #### define noise & denoise functions
    def denoise_step(params, x, mask, time, rope_index, rng_key):
        time = jnp.full((x.shape[0],), time) ## (dbs,)
        eps_pred = dit_net.apply(
            {'params': params['params']['net']}, x, mask, 
            time, tokens_rope_index = rope_index) ## (dbs, npt, d)
        mean, variance, log_variance = scheduler.p_mean_variance(
            x, time, eps_pred, clamp_x0_fn = None, clip = False)
        rng_key, sub_key = jax.random.split(rng_key)
        x = mean + jnp.exp(0.5 * log_variance) * jax.random.normal(sub_key, x.shape)
        return x, rng_key
    jit_denoise_step = jax.jit(denoise_step)

    def noise_denoise(x, T, rng_key, cached):
        ## args
        params = cached['params']
        mask = cached['mask']
        eq_steps = cached['eq_steps']
        rope_index = cached['rope_index']
        ## noise
        x, rng_key = jit_noise(x, T, rng_key)
        ## denoise
        for t_d in tqdm(range(T)):
            t = T - t_d
            for _ in range(eq_steps):
                x, rng_key = jit_denoise_step(params, x, mask, t, rope_index, rng_key)
                x, rng_key = jit_noise_step(x, t, rng_key)
            x, rng_key = jit_denoise_step(params, x, mask, t, rope_index, rng_key)
        return x
    
    #### inference
    cached = {
        'params': params,
        'mask': jnp.ones((dbs, npt), dtype = jnp.int32),
        'rope_index': jnp.array(
            [np.arange(npt),] * (dbs),dtype = jnp.int32).reshape(dbs, npt),
        'eq_steps': 10,
    }
    rng_key = jax.random.PRNGKey(args.random_seed)
    results = []
    print("----------------------------------------------")
    for _t in range(0, 200, 10):
        print(f"T: {_t}")
        rng_key, this_key = jax.random.split(rng_key)
        _x = encoder_function(test_molecules['graphs'])
        _x *= jnp.sqrt(_x.shape[-1]) ## scale
        _x = noise_denoise(_x, _t, this_key, cached)
        output_smiles = decoder_function(_x,)['smiles']
        results.append(output_smiles)
        print("----------------------------------------------")

    ### save
    with open(args.save_path, 'wb') as f:
        pkl.dump(results, f)

if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, required=True)
    parser.add_argument('--params_path', type=str, required=True)
    parser.add_argument('--vae_config_path', type=str, required=True)
    parser.add_argument('--vae_params_path', type=str, required=True)
    parser.add_argument('--alphabet_path', type=str, required=True)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--sampling_method', type=str, default='beam')
    parser.add_argument('--beam_size', type=int, default=4)
    parser.add_argument('--init_molecule_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    main(args)
    