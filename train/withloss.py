import os
import sys
import math
import numpy as np
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import flax.linen as nn

from functools import partial
from jax import Array
from jax import lax
from ml_collections.config_dict import ConfigDict
from typing import Any

from .utils import cosine_const_schedule, exp_warmup_const_schedule

def softmax_cross_entropy(logits, label, seq_mask):
    ### logits: (Nres, Nbins)
    ### label: (Nres, Nbins) (one hot)
    ### seq_mask: (Nres) 
    
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(log_probs * label, axis=-1)
    
    return jnp.sum(loss * seq_mask, axis=-1) # (BK, N) -> (BK,)

def gaussian_kernel_mmd(x, y, sigma):

    def _gaussian_kernel(x, y, sigma = sigma):
        ## x: (B, F) y: (B, F)
        _dim = x.shape[-1]
        x = x[:, None]
        y = y[None, :]
        return jnp.exp((-1) * jnp.sum((x - y) ** 2, axis=-1) / (sigma ** 2)) # (B, B)

    k_xx = _gaussian_kernel(x, x)
    k_yy = _gaussian_kernel(y, y)
    k_xy = _gaussian_kernel(x, y)
    return jnp.mean(k_xx + k_yy - 2 * k_xy) # (B, B) -> (,)

def info_nce_function(f, pair_f, candidate_fs, mask, beta = 1.):
    ### f/pair_f: (B, 1, F), candidate_fs: (B, N, F), mask: (B, N)
    ### return: (B,)

    def _dist(x, y): ## cosine distance
        ### Higher is better!
        return jnp.sum(x * y, -1) / (jnp.linalg.norm(x, axis=-1) * jnp.linalg.norm(y, axis=-1) + 1e-6) # (B, N)

    distance_target = _dist(f, pair_f) # (B, 1,)
    distance_candidates = _dist(f, candidate_fs) # (B, N,)
    neg_logits = (distance_candidates - distance_target)
    neg_logits *= jax.lax.stop_gradient(beta)
    return jnp.mean(nn.logsumexp(neg_logits + mask * (-5e4), -1)) # (,)

def align_loss_function(x, y, alpha = 2., beta = 1.):
    ### x / y: (b, Q, D)
    # breakpoint() ## check here
    b, q, d = x.shape
    dist_ = jnp.sum((x - y) ** 2, axis = 2) ## (b, Q)
    dist_ = jnp.power(dist_, alpha / 2) ## (b, Q)
    # dist_ = jax.nn.logsumexp(dist_ * beta, axis = 1) / q / beta ## (b,)
    dist_ = jnp.mean(dist_, axis = 1) ## (b,) ### mean mode deprecated
    # dist_ = jnp.max(dist_, axis = 1) ## (b,)
    return jnp.mean(dist_, axis = 0) ## (,)

def uni_loss_function(x, t = 2., beta = 1.):
    ### x: (b, Q, D)
    ### we calculate the uniformity loss of groups first and reduce mean
    b, q, d = x.shape
    x = jnp.swapaxes(x, 0, 1) ## (Q, b, D)
    ## (Q, b, 1, D) - (Q, 1, b, D) -> (Q, b, b, D) -> (Q, b, b)
    sq_dist = jnp.sum((x[:, :, None] - x[:, None]) ** 2, axis = 3) ## (Q, b, b)
    # breakpoint() ## check here
    sq_dist = jnp.reshape(sq_dist, (q, b*b)) ## (Q, b*b)
    # loss = jax.nn.logsumexp((-t) * sq_dist, axis = -1) - jnp.log(b * b) ## (Q,)
    # loss = jax.nn.logsumexp(loss * beta, axis = 0) / q / beta ## (,)
    loss = jax.nn.logsumexp((-t) * sq_dist,) - jnp.log(b * b * q) ## (,)
    return loss

### simCLR
def align_loss_function_sr(x, y, alpha = 2.):
    ### x / y: (b, f)
    b, f = x.shape
    dist_ = jnp.sum((x - y) ** 2, axis = 1) ## (b,)
    dist_ = jnp.power(dist_, alpha / 2) ## (b,)
    return jnp.mean(dist_, axis = 0) ## (,)

def uni_loss_function_sr(x, t = 2.):
    ### x: (b, f)
    # breakpoint() ## check here
    b, qd = x.shape
    ### (b, 1, f) - (1, b, f) -> (b, b, f) -> (b, b)
    sq_dist = jnp.sum((x[:, None] - x[None, ...]) ** 2, axis = 2)
    ### (b, b) -> (b*b,) -> (,)
    sq_dist = jnp.reshape(sq_dist, (b*b,))
    return jax.nn.logsumexp((-t) * sq_dist,) - jnp.log(b * b) ## (,)

class L2SeqGenWithLoss(nn.Module):

    train_config: ConfigDict
    global_config: ConfigDict
    generator: nn.Module
    pmap_flag: bool = True

    def setup(self):

        self._dtype = jnp.float16 if self.global_config.bf16_flag else jnp.float32

        settings_config = self.train_config.settings
        self.vocab_size = settings_config.vocab_size ## vocab size include BOS/EOS/UNK/SEP...
        self.num_prefix = settings_config.num_prefix_tokens
        self.num_sink = settings_config.num_sink_tokens
        self.loss_weights = self.train_config.loss_weights

    def __call__(self, input_features, labels, step_it = 0):

        # ## define gamma schedule function
        # gamma_config = self.train_config.gamma
        # gamma_schedule = partial(
        #     exp_warmup_const_schedule, min_val=gamma_config['min'],
        #     max_val=gamma_config['max'], warmup_steps=gamma_config['warmup_steps'],
        # )
        # gamma = gamma_schedule(step_it)

        ## define loss weight schedule function ### 24-08-12 LB1 deprecated now

        ## get logits
        seq_logits, aux = self.generator(input_features)
        ## cast to fp32
        seq_logits, seq_mask, seq_label = jax.tree_util.tree_map(
            jnp.float32, (seq_logits, labels['mask'], labels['label'])
        )
        ## reshape
        batch_size, num_k, n_seq = seq_mask.shape
        seq_mask, seq_label = jax.tree_util.tree_map(
            lambda f: f.reshape(batch_size*num_k, n_seq), (seq_mask, seq_label)
        ) # (BK, N), (BK, N)

        #### get reconstruction loss
        n_prefix = self.num_prefix
        seq_label = jax.nn.one_hot(
            seq_label, self.vocab_size, dtype=jnp.float32, axis=-1,
        )
        log_p_unmasked = (-1) * softmax_cross_entropy(seq_logits, seq_label, seq_mask) # (BK,)
        log_p_unmasked = log_p_unmasked.reshape(batch_size, num_k) # (B, K)
        mask_p = jnp.reshape(seq_mask.sum(-1) > 0, (batch_size, num_k))
        k = jnp.sum(mask_p, -1) # (B,)
        recon_loss = jnp.sum(log_p_unmasked, -1) / (k * (n_seq - n_prefix)) # (B,)
        recon_loss = jnp.mean(recon_loss) * (-1) # (,)

        # #### 11-16 update: alignment/uniformity loss with simCLR
        # latent_feat = aux['sim_feat'] ## (B, F)
        # latent_feat = jnp.float32(latent_feat)
        # latent_feat_x, latent_feat_y = jnp.split(latent_feat, 2, axis = 0) # (B, F) -> (B/2, F)
        # ## gather from processes
        # ## WARNING: in this code all processes should have the same sample distribution: b/2 for cluster center and b/2 for neighbor
        # ## NO MASK HERE
        # if self.pmap_flag:
        #     latent_feat_x = jax.lax.all_gather(latent_feat_x, axis_name = 'i', tiled = True) # (B/2*N, F), N = n_global_devices
        #     latent_feat_y = jax.lax.all_gather(latent_feat_y, axis_name = 'i', tiled = True)
        # # breakpoint() ## check shape here
        # cts_config = self.train_config.contrastive
        # align_loss = align_loss_function_sr(latent_feat_x, latent_feat_y, alpha = cts_config.alpha) # (,) sr for simCLR
        # uni_loss = 0.5 * (
        #     uni_loss_function_sr(latent_feat_x, t = cts_config.t) + uni_loss_function_sr(latent_feat_y, t = cts_config.t)
        # ) # (,)

        loss = recon_loss * self.loss_weights['reconstruct'] ## only reconstruction loss
        loss_dict = {
            'loss': loss,
            'reconstruct': recon_loss,}

        return loss, loss_dict

class DecoderWithLoss(nn.Module):

    train_config: ConfigDict
    global_config: ConfigDict
    generator: nn.Module

    def setup(self):

        self._dtype = jnp.float16 if self.global_config.bf16_flag else jnp.float32

        settings_config = self.train_config.settings
        self.vocab_size = settings_config.vocab_size ## vocab size include BOS/EOS/UNK/SEP...
        self.num_prefix = settings_config.num_prefix_tokens
        self.num_sink = settings_config.num_sink_tokens
        self.loss_weights = self.train_config.loss_weights

    def __call__(self, input_features, labels, step_it = 0):

        ## get logits
        seq_logits = self.generator(input_features['graph_feat'], input_features['seq_feat'])
        ## cast to fp32
        seq_logits, seq_mask, seq_label = jax.tree_util.tree_map(
            jnp.float32, (seq_logits, labels['mask'], labels['label'])
        )
        ## reshape
        batch_size, num_k, n_seq = seq_mask.shape
        seq_mask, seq_label = jax.tree_util.tree_map(
            lambda f: f.reshape(batch_size*num_k, n_seq), (seq_mask, seq_label)
        ) # (BK, N), (BK, N)

        #### get reconstruction loss
        n_prefix = self.num_prefix
        seq_label = jax.nn.one_hot(
            seq_label, self.vocab_size, dtype=jnp.float32, axis=-1,
        )
        log_p_unmasked = (-1) * softmax_cross_entropy(seq_logits, seq_label, seq_mask) # (BK,)
        log_p_unmasked = log_p_unmasked.reshape(batch_size, num_k) # (B, K)
        mask_p = jnp.reshape(seq_mask.sum(-1) > 0, (batch_size, num_k))
        k = jnp.sum(mask_p, -1) # (B,)
        recon_loss = jnp.sum(log_p_unmasked, -1) / (k * (n_seq - n_prefix)) # (B,)
        recon_loss = jnp.mean(recon_loss) * (-1) # (,)

        loss = recon_loss * self.loss_weights['reconstruct']
        loss_dict = {
            'loss': loss,
            'reconstruct': recon_loss,
        }

        return loss, loss_dict

class MMDSeqGenWithLoss(nn.Module):

    train_config: ConfigDict
    global_config: ConfigDict
    generator: nn.Module
    pmap_flag: bool = True

    def setup(self):

        self._dtype = jnp.float16 if self.global_config.bf16_flag else jnp.float32

        settings_config = self.train_config.settings
        self.vocab_size = settings_config.vocab_size ## vocab size include BOS/EOS/UNK/SEP...
        self.num_prefix = settings_config.num_prefix_tokens
        self.num_sink = settings_config.num_sink_tokens
        self.loss_weights = self.train_config.loss_weights

    def __call__(self, input_features, labels, step_it = 0):

        ## get logits
        seq_logits, aux = self.generator(input_features)
        ## cast to fp32
        seq_logits, seq_mask, seq_label = jax.tree_util.tree_map(
            jnp.float32, (seq_logits, labels['mask'], labels['label'])
        )
        ## reshape
        batch_size, num_k, n_seq = seq_mask.shape
        seq_mask, seq_label = jax.tree_util.tree_map(
            lambda f: f.reshape(batch_size*num_k, n_seq), (seq_mask, seq_label)
        ) # (BK, N), (BK, N)

        ######### get reconstruction loss #########
        n_prefix = self.num_prefix
        seq_label = jax.nn.one_hot(
            seq_label, self.vocab_size, dtype=jnp.float32, axis=-1,
        )
        log_p_unmasked = (-1) * softmax_cross_entropy(seq_logits, seq_label, seq_mask) # (BK,)
        log_p_unmasked = log_p_unmasked.reshape(batch_size, num_k) # (B, K)
        mask_p = jnp.reshape(seq_mask.sum(-1) > 0, (batch_size, num_k))
        k = jnp.sum(mask_p, -1) # (B,)
        recon_loss = jnp.sum(log_p_unmasked, -1) / (k * (n_seq - n_prefix)) # (B,)
        recon_loss = jnp.mean(recon_loss) * (-1) # (,)

        ######### alignment / uniformity loss (not on hypersphere!) #########
        latent_feat = aux['graph_feat'] # (B, Q, D)
        latent_feat = jnp.float32(latent_feat)
        latent_feat_x, latent_feat_y = jnp.split(latent_feat, 2, axis = 0) # (B, Q, D) -> (B/2, Q, D)
        ## gather from processes
        ## WARNING: in this code all processes should have the same sample distribution: b/2 for cluster center and b/2 for neighbor
        ## NO MASK HERE
        if self.pmap_flag:
            latent_feat_x = jax.lax.all_gather(latent_feat_x, axis_name = 'i', tiled = True) # (B/2*N, Q, D), N = n_global_devices
            latent_feat_y = jax.lax.all_gather(latent_feat_y, axis_name = 'i', tiled = True)
        # breakpoint() ## check shape here
        cts_config = self.train_config.contrastive
        align_loss = align_loss_function(latent_feat_x, latent_feat_y, alpha = cts_config.alpha) # (,)
        uni_loss = 0.5 * (
            uni_loss_function(latent_feat_x, t = cts_config.t) + uni_loss_function(latent_feat_y, t = cts_config.t)
        ) # (,)
        
        ######### get mmd regularization loss #########
        mmd_loss = 0.
        latent_feat = aux['graph_feat'].reshape(batch_size, -1) # (B, Q, D) -> (B, QD)
        latent_feat = jnp.float32(latent_feat)
        gaussian_feat = jax.random.normal(
            key=self.make_rng('latent'), shape=latent_feat.shape, dtype=latent_feat.dtype,
        )
        for _sigma in self.train_config.mmd_sigma:
            mmd_loss += gaussian_kernel_mmd(latent_feat, gaussian_feat, _sigma)

        loss = recon_loss * self.loss_weights['reconstruct'] + \
            mmd_loss * self.loss_weights['mmd'] + \
            align_loss * self.loss_weights['align'] + \
            uni_loss * self.loss_weights['uniformity'] ### the lambda in paper
        loss_dict = {
            'loss': loss,
            'reconstruct': recon_loss,
            'align': align_loss,
            'uniformity': uni_loss,
            'mmd': mmd_loss,
        }

        return loss, loss_dict

class DiTWithLoss(nn.Module):
    
    train_config: ConfigDict
    global_config: ConfigDict
    net: nn.Module
    scheduler: Any
    pmap_flag: bool = True

    @nn.compact
    def __call__(self, input_dict):
        #### feat: (bs, npt, d)

        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        #### noise & denoise
        x_0 = input_dict['feat']
        bs, npt, d = x_0.shape
        time_key = self.make_rng('time_key')
        time_steps = self.train_config.diffusion_timesteps
        random_t = jax.random.randint(time_key, (bs,), 0, time_steps)
        normal_key = self.make_rng('normal_key')
        eps = jax.random.normal(normal_key, x_0.shape, dtype = jnp.float32)
        x_t = self.scheduler.q_sample(x_0, random_t, eps)

        #### run
        x_t, random_t = jtu.tree_map(
            lambda arr: arr.astype(arr_dtype), (x_t, random_t),
        )
        eps_pred = self.net(
            x_t, jnp.ones((bs, npt), dtype = arr_dtype), random_t,
            tokens_rope_index = jnp.arange(npt, dtype = jnp.int32)[None, :].repeat(bs, axis = 0),
        )
        eps_pred = jnp.float32(eps_pred)

        #### L2 Loss
        loss = (eps_pred - eps) ** 2 # (bs, npt, d)
        loss = jnp.mean(jnp.mean(loss, axis = 2)) # (,)

        return loss
