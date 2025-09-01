import jax.numpy as jnp
import flax.linen as nn
from ml_collections import ConfigDict

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.common.utils import safe_l2_normalize
from src.common.layers.mlp import MLP

class Projector(nn.Module):
    """for predictor with no pocket encoder"""

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, latents,):
        # latents: (dbs, npt, d) NOTE: scaled(norm=d).
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        dbs, npt, d = latents.shape
        latents = safe_l2_normalize(latents / jnp.sqrt(d), axis=-1)
        # mlp_net = MLP(
        #     output_sizes=self.config.output_sizes,
        #     activation=self.config.activation,
        #     dtype=arr_dtype,
        #     dropout_rate=self.config.dropout_rate,
        #     dropout_flag=self.global_config.dropout_flag,
        # )
        # mlp_net_lists = [mlp_net,] * self.config.n_scores
        latents = jnp.mean(latents, axis=-2) # (dbs, d)
        # outputs = [mlp_net_lists[i](latents) for i in range(self.config.n_scores)]
        outputs = []
        for i in range(self.config.n_scores):
            mlp_net = MLP(
                output_sizes=self.config.output_sizes,
                activation=self.config.activation,
                dtype=arr_dtype,
                dropout_rate=self.config.dropout_rate,
                dropout_flag=self.global_config.dropout_flag,
                name=f'out_projector_{i}',
            )
            outputs.append(mlp_net(latents))
        outputs = jnp.concatenate(outputs, axis=-1) # (dbs, n_scores)
        return outputs * self.config.scale + self.config.shift

def cosine_cutoff(distance, mask, cutoff):
    decay = 0.5 * (1 + jnp.cos(jnp.pi * distance / cutoff))
    mask = jnp.logical_and(mask, distance < cutoff)
    decay = jnp.where(mask, decay, 0)
    return decay, mask

import functools
from jax import Array
from src.common.rbf.gaussian import GaussianBasis
def segment_mask(
    q_segment_ids: Array,
    kv_segment_ids: Array,
):
    """From jax.experimental.pallas.ops.attention"""
    # [B, T, 1] or [T, 1]
    q_segment_ids = jnp.expand_dims(q_segment_ids, axis=-1)
    # [B, 1, S] or [1, S]
    if kv_segment_ids.ndim == 1:
        kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=0)
    else:
        kv_segment_ids = jnp.expand_dims(kv_segment_ids, axis=1)
    return jnp.equal(q_segment_ids, kv_segment_ids).astype(jnp.bool_)

class PairEmbedding(nn.Module):
    """Embedding 3D features."""

    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, d_ij: Array, m_i: Array):
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        cutoff_fn = functools.partial(cosine_cutoff, cutoff=self.config.cutoff)
        rbf_fn = GaussianBasis(self.config.rbf, self.global_config)
        
        bs_, a_, a_ = d_ij.shape
        m_ii = jnp.eye(a_, dtype=arr_dtype)
        d_ij = jnp.where(m_ii, self.config.d_self, d_ij)
        # (B, A, A,) -> (B, A, A, F)
        g_ij = rbf_fn(d_ij)
        if self.config.proj_flag:
            # (B, A, A, F) -> (B, A, A, D)
            g_ij = nn.Dense(
                features=self.config.proj_dim,
                use_bias=False,
                name="g_proj",
                param_dtype=jnp.float32,
                dtype=arr_dtype,
            )(g_ij)
        # (B, A, A,)
        m_ij = segment_mask(m_i, m_i)
        c_ij, _ = cutoff_fn(d_ij, m_ij)

        return g_ij, c_ij

from src.model.transformer import ResiDualTransformerBlock
from src.module.transformer import NormBlock
class PocketEncoder(nn.Module):
    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, pocket_features):
        """pocket_features: {
            'esm_embedding': (dbs, nseq, d'),
            'residue_mask': (dbs, nseq),
            'distance_matrix': (dbs, nseq, nseq),
            }
        """
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        single_act = nn.Dense(
            features = self.config.hidden_size,
            dtype = arr_dtype,
            use_bias = False,
        )(pocket_features['esm_embedding'])
        single_mask = jnp.int32(pocket_features['residue_mask'])
        pair_act, pair_cutoff = PairEmbedding(
            config = self.config.pair_embedding,
            global_config = self.global_config,
        )(pocket_features['distance_matrix'], single_mask)
        pair_act = pair_act * pair_cutoff[..., None] # (dbs, nseq, nseq, f)

        single_act, acc_single_act = single_act, single_act
        if self.config.shared_weights:
            _residual_transformer_block = ResiDualTransformerBlock(
                    config = self.config.transformer_block,
                    global_config = self.global_config,)
            for _ in range(self.config.n_layers):
                single_act, acc_single_act = \
                    _residual_transformer_block(s_i=single_act, acc_s_i=acc_single_act, m_i=single_mask, z_ij=pair_act,)
        else:
            for _ in range(self.config.n_layers):
                single_act, acc_single_act = ResiDualTransformerBlock(
                    config = self.config.transformer_block,
                    global_config = self.global_config,
                )(s_i=single_act, acc_s_i=acc_single_act, m_i=single_mask, z_ij=pair_act,)
        single_act += NormBlock(
            norm_method = self.global_config.norm_method,
            eps = self.global_config.norm_small,
        )(acc_single_act)

        return single_act # (dbs, nseq, d)

from src.model.qformer import QFormerBlock
class FusionEncoder(nn.Module):
    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, mol_act, pocket_act, pocket_mask,):

        mol_act, acc_mol_act = mol_act, mol_act
        bs, npt, _ = mol_act.shape
        mol_mask = jnp.ones((bs, npt), dtype=jnp.int32) # (dbs, npt)

        if self.config.shared_weights:
            _qformer_block = QFormerBlock(
                config = self.config.qformer_block,
                global_config = self.global_config,
            )
            for _ in range(self.config.n_layers):
                mol_act, acc_mol_act = _qformer_block(
                    acc_mol_act, mol_act, pocket_act, mol_mask, pocket_mask,)
        else:
            for _ in range(self.config.n_layers):
                mol_act, acc_mol_act = QFormerBlock(
                    config = self.config.qformer_block,
                    global_config = self.global_config,
                )(acc_mol_act, mol_act, pocket_act, mol_mask, pocket_mask,)
        mol_act += NormBlock(
            norm_method = self.global_config.norm_method,
            eps = self.global_config.norm_small,
        )(acc_mol_act)
        return mol_act # (dbs, npt, f)

class PretrainPredictorWithPocket(nn.Module):
    """For pretraining
    """
    encoder_config: ConfigDict
    projector_config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, input_data):
        # latents: (dbs, npt, d) NOTE: scaled(norm=d).

        latents = input_data.pop('latent_embedding') # (dbs, npt, d)
        pocket_features = input_data
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        dbs, npt, d = latents.shape
        latents = safe_l2_normalize(latents / jnp.sqrt(d), axis=-1)

        mol_embedding = nn.Dense(
            features = self.encoder_config.hidden_size,
            dtype = arr_dtype,
            use_bias = False,
            name = 'mol_embedding',
        )(latents) # (dbs, npt, f)

        pocket_encoder = PocketEncoder(
            config = self.encoder_config.pocket_encoder,
            global_config = self.global_config,
            name = 'pocket_encoder',
        ) # shared weights
        fusion_encoder = FusionEncoder(
            config = self.encoder_config.fusion_encoder,
            global_config = self.global_config,
            name = 'fusion_encoder',
        )
        pocket_embedding = pocket_encoder(pocket_features) # (dbs, nseq, f)
        # cross attention
        mol_embedding = fusion_encoder(mol_embedding, pocket_embedding, input_data['residue_mask']) # (dbs, npt, f)

        # project out
        mlp_net = MLP(
            output_sizes = self.projector_config.output_sizes,
            activation = self.projector_config.activation,
            dtype = arr_dtype,
            dropout_rate = self.projector_config.dropout_rate,
            dropout_flag = self.global_config.dropout_flag,
            name = 'out_projector'
        ) # separate weights
        mol_embedding = jnp.mean(mol_embedding, axis=-2) # (dbs, f)
        return mlp_net(mol_embedding) * self.projector_config.scale + self.projector_config.shift

from jax.tree_util import tree_map
class EncoderWithPocket(nn.Module):
    """For downstream tasks, while params and pocket features are frozen.
    """
    encoder_config: ConfigDict
    global_config: ConfigDict
    pocket_features: dict

    @nn.compact
    def __call__(self, latents,):
        # dbs: input_batch_size, npoks: number of pockets,
        # latents: (dbs, npt, d) NOTE: scaled(norm=d).

        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        dbs, npt, d = latents.shape
        npoks = self.pocket_features['residue_mask'].shape[0]
        latents = safe_l2_normalize(latents / jnp.sqrt(d), axis=-1)
        f = self.encoder_config.hidden_size

        mol_embedding = nn.Dense(
            features = self.encoder_config.hidden_size,
            dtype = arr_dtype,
            use_bias = False,
            name = 'mol_embedding',
        )(latents) # (dbs, npt, f)

        pocket_encoder = PocketEncoder(
            config = self.encoder_config.pocket_encoder,
            global_config = self.global_config,
            name = 'pocket_encoder',
        )
        fusion_encoder = FusionEncoder(
            config = self.encoder_config.fusion_encoder,
            global_config = self.global_config,
            name = 'fusion_encoder',
        )

        # (npoks, ...) -> (dbs*npoks, ...)
        # TODO: We can infer the pocket embedding in advance
        pocket_embedding = pocket_encoder(self.pocket_features) # (dbs, nseq, f)
        _, nres, _ = pocket_embedding.shape
        pocket_embedding = pocket_embedding[None, :].repeat(dbs, axis=0).reshape((dbs*npoks, nres, f))
        residue_mask = self.pocket_features['residue_mask'][None, :].repeat(dbs, axis=0).reshape((dbs*npoks, nres))

        # cross attention
        # (dbs, npt, f) -> (dbs*npoks, npt, f)
        mol_embedding = jnp.repeat(mol_embedding[:, None], npoks, axis=1).reshape((dbs*npoks, npt, f))
        mol_embedding = fusion_encoder(mol_embedding, pocket_embedding, residue_mask) # (dbs*npoks, npt, f)
        return mol_embedding.reshape((dbs, npoks, npt, f))

class ProjectorWithPocket(nn.Module):
    config: ConfigDict
    global_config: ConfigDict

    @nn.compact
    def __call__(self, emb):
        # (dbs, npoks, npt, f)
        arr_dtype = jnp.bfloat16 if self.global_config.bf16_flag else jnp.float32
        dbs, npoks, npt, f = emb.shape
        # mlp_net = MLP(
        #     output_sizes=self.config.output_sizes,
        #     activation=self.config.activation,
        #     dtype=arr_dtype,
        #     dropout_rate=self.config.dropout_rate,
        #     dropout_flag=self.global_config.dropout_flag,
        #     # name = 'out_projector',
        # )
        # mlp_net_lists = [mlp_net,] * npoks
        emb = jnp.mean(emb, axis=-2) # (dbs, npoks, f)
        outputs = []
        for i in range(npoks):
            mlp_net = MLP(
                output_sizes=self.config.output_sizes,
                activation=self.config.activation,
                dtype=arr_dtype,
                dropout_rate=self.config.dropout_rate,
                dropout_flag=self.global_config.dropout_flag,
                name=f'out_projector_{i}',
            )
            outputs.append(mlp_net(emb[:, i]))
        # outputs = [mlp_net_lists[i](emb[:, i]) for i in range(self.config.n_scores)] # [(dbs, f), ...] -> [(dbs, 1), ...]
        outputs = jnp.concatenate(outputs, axis=-1) # (dbs, n_scores)
        return outputs * self.config.scale + self.config.shift

def set_encoder_config(
        hidden_size: int,
        n_head: int,
        n_pocket_encoder_layers: int,
        n_fusion_encoder_layers: int,
        dropout_rate: float = 0.05,
        shared_weights: bool = True,
        ):
    
    pocket_encoder_config = {
        'hidden_size': hidden_size,
        'pair_embedding': {
            'cutoff': 20., # angstrom, TODO: need to check
            'rbf': {
                'r_max': 20.,
                'r_min': 1.,
                'num_basis': 64,
                'sigma': 0.25,
                'delta': None,
                'clip_distance': False,
            },
            'd_self': 0.1, # angstrom,
            'proj_flag': True,
            'proj_dim': hidden_size,
        },
        'n_layers': n_pocket_encoder_layers,
        'shared_weights': shared_weights,
        'transformer_block': {
            'dropout_rate': dropout_rate,
            'attention_embedding': {
                "attention_type": "self",
                "dim_feature": hidden_size,
                "n_head": n_head,
                "embedding_pair_flag": True,
                "kernel_initializer": "glorot_uniform",
            },
            'hyper_attention_flag': False,
            'attention_kernel': {
                "attention_type": "self",
                "flash_attention_flag": False,
                "has_bias": True,
                "causal_flag": False,
            },
            "post_attention": {
                "out_dim": hidden_size,
                "gating_flag": True,
            },
            'transition': {
                "method": "glu", ## glu
                "transition_factor": 2,
                "kernel_initializer": "xavier_uniform",
                "act_fn": "gelu",
            },
        }
    }

    fusion_encoder_config = {
        'n_layers': n_fusion_encoder_layers,
        'shared_weights': shared_weights,
        'qformer_block': {
            'self_attention': {
                "attention_embedding": {
                    "attention_type": "self",
                    "dim_feature": hidden_size,
                    "n_head": n_head,
                    "embedding_pair_flag": False,
                    "kernel_initializer": "glorot_uniform",
                },

                "hyper_attention_flag": False,
                
                "attention_kernel": {
                    "attention_type": "self",
                    "flash_attention_flag": False,
                    "has_bias": False,
                    "causal_flag": False,
                },

                "post_attention": {
                    "out_dim": hidden_size,
                    "gating_flag": True,
                },
                "dropout_rate": dropout_rate,
            },
            'cross_attention': {
                "attention_embedding": {
                    "attention_type": "cross",
                    "dim_feature": hidden_size,
                    "n_head": n_head,
                    "embedding_pair_flag": False,
                    "kernel_initializer": "glorot_uniform",
                },

                "hyper_attention_flag": False,
                
                "attention_kernel": {
                    "attention_type": "cross",
                    "flash_attention_flag": False,
                    "has_bias": False,
                    "causal_flag": False,
                },

                "post_attention": {
                    "out_dim": hidden_size,
                    "gating_flag": True,
                },
                "dropout_rate": dropout_rate,
            },
            'transition_block': {
                'transition': {
                    "method": "glu", ## glu
                    "transition_factor": 2,
                    "kernel_initializer": "xavier_uniform",
                    "act_fn": "gelu",
                },
                'dropout_rate': dropout_rate,
            },
        },
    }

    encoder_config = {
        'hidden_size': hidden_size,
        'pocket_encoder': pocket_encoder_config,
        'fusion_encoder': fusion_encoder_config,
    }

    return ConfigDict(encoder_config)

def set_projector_config(
        input_dim: int, scale: float = 1.0, shift: float = 0.0, dropout_rate: float = 0.05,
        n_scores: int = 1, # for infer projector
        ):
    projector_config = {
        'output_sizes': [input_dim*2, input_dim, 1],
        'activation': 'leaky_relu',
        'scale': scale,
        'shift': shift,
        'dropout_rate': dropout_rate,
        'n_scores': n_scores,
    }
    return ConfigDict(projector_config)

def set_global_config(
        norm_method: str = 'layernorm',
        norm_small: float = 1e-6,
        bf16_flag: bool = True,
        dropout_flag: bool = True,
):
    global_config = {
        'norm_method': norm_method,
        'norm_small': norm_small,
        'bf16_flag': bf16_flag,
        'dropout_flag': dropout_flag,
    }
    return ConfigDict(global_config)
