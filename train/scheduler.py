import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import flax.linen as nn

from ml_collections import ConfigDict
from typing import Any

def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = arr[timesteps]
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res

class GaussianDiffusion:
    def __init__(self, config):
        num_diffusion_timesteps = config.diffusion_timesteps
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        betas = jnp.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=jnp.float32)
        self.betas = betas
        self.sqrt_betas = jnp.sqrt(self.betas)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas = alphas
        self.sqrt_alphas = jnp.sqrt(self.alphas)
        self.alphas_cumprod = jnp.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = jnp.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = jnp.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = jnp.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = jnp.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = jnp.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = jnp.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = jnp.log(
            jnp.append(self.posterior_variance[1], self.posterior_variance[1:])
        ) if len(self.posterior_variance) > 1 else jnp.array([])

        # (beta_t * atm1.sqrt()) / (1.0 - at)
        self.posterior_mean_coef1 = (
            betas * jnp.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # ((1 - atm1) * (1 - beta_t).sqrt()) / (1.0 - at)
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * jnp.sqrt(alphas) / (1.0 - self.alphas_cumprod)
        )
        
        # ### info nce bgm kernel
        # if config.use_info_nce: 
        #     num_infonce_timesteps = config.info_nce.time_steps
        #     ## linear decay
        #     self.infonce_beta = jnp.linspace(config.info_nce.beta_max, 0.0, num_infonce_timesteps, dtype=jnp.float32)
        #     self.infonce_beta = jnp.pad(self.infonce_beta, (0, self.num_timesteps - num_infonce_timesteps), 
        #                                 mode='constant', constant_values=0.0)
        
    def alphas_cumprod_to_t(self, alphas_cumprod_val):
        alphas_cumprod_val = jnp.clip(alphas_cumprod_val, 
                                      jnp.min(self.alphas_cumprod)+1e-6, 
                                      jnp.max(self.alphas_cumprod)-1e-6)
        
        index = jnp.searchsorted(self.alphas_cumprod[::-1], alphas_cumprod_val)
        index = self.num_timesteps - index
        
        return index - 1 + (alphas_cumprod_val - self.alphas_cumprod[index - 1]) / (self.alphas_cumprod[index] - self.alphas_cumprod[index - 1])
        
    
    def q_sample(self, x_start, t, eps): # q(x_t | x_0)
        """ q(x_t | x_0) """
        assert eps.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * eps
        )
        
    def q_sample_step(self, x, t, noise): # q(x_t | x_{t-1})
        """ q(x_t | x_{t-1}) = N(x_t | sqrt(1 - beta_t) * x_{t-1}, beta_t I))) """
        return (
            _extract_into_tensor(self.sqrt_alphas, t, x.shape) * x
            + _extract_into_tensor(self.sqrt_betas, t, x.shape) * noise
        )
    
    def _predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def p_mean_variance(self, x_t, t, eps, clamp_x0_fn=None, clip=True):
        """p(x_{t-1} | x_t) """
        # TODO: Handle learned variance.
        
        model_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        model_log_variance = _extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        pred_xstart = self._predict_xstart_from_eps(x_t=x_t, t=t, eps=eps)
        pred_xstart_clipped = jnp.clip(pred_xstart, -1, 1)
        pred_xstart = jnp.where(clip, pred_xstart_clipped, pred_xstart)
        if clamp_x0_fn:
            pred_xstart = clamp_x0_fn(pred_xstart)

        model_mean = _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * pred_xstart \
                        + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        return model_mean, model_variance, model_log_variance

    def p_ddim(self, x_t, t, eps):
        at = _extract_into_tensor(self.alphas_cumprod, t, x_t.shape)
        at_next = _extract_into_tensor(self.alphas_cumprod_prev, t, x_t.shape)
        c2 = ((1 - at_next)).sqrt()
        x0_t = (x_t - eps * (1 - at).sqrt()) / at.sqrt()
        xt_next = at_next.sqrt() * x0_t + c2 * eps
        return xt_next