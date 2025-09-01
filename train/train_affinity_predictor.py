
import os
import jax
import numpy as np
import pandas as pd
import jax.numpy as jnp
import optax
import pickle as pkl
import copy
import datetime
import tensorflow as tf

from flax import linen as nn
from jax.tree_util import tree_map
from logging import Logger
from ml_collections import ConfigDict
from .utils import pmean_tree

Optimizer = optax.GradientTransformation
class Trainer:
    """pretraining for affinity predictor"""
    def __init__(
            self,
            config: ConfigDict, # train config
            net: nn.Module,
            optimizer: Optimizer,
            recoder: Logger,
            name_list: list,
            # data_root: str,
            save_root: str,
            pmap_flag: bool = False,
            ):
        
        self.config = config
        self.net = net
        self.eval_net: nn.Module = net
        self.eval_net.global_config.dropout_flag = False
        self.optimizer = optimizer
        self.recoder = recoder
        # self.split_rate = config.split_rate
        self.save_root = save_root
        self.pmap_flag = pmap_flag
        self.name_list = name_list[1:] ### a list of dirs
        os.makedirs(save_root, exist_ok=True)
        # if exists, raise warning
        if os.path.exists(save_root):
            self.recoder.warning(f'Save root {save_root} already exists, will overwrite it !')

        """
            Load full crossdocked data.
        """
        
        _fn = name_list[0]
        with open(_fn, 'rb') as f:
            _data = pkl.load(f)
            _, _n, _d = _data['latent_embedding'].shape
            _, _nres, _desm = _data['esm_embedding'].shape
        assert np.allclose(np.sum(_data['latent_embedding'] ** 2, axis=-1), _d, rtol=0.05)
        self.init_data = _data
        self.train_data = {
            'latent_embedding': np.empty((0, 16, 32), dtype = np.float32),
            'residue_mask': np.empty((0, _nres,), dtype = np.int32),
            'esm_embedding': np.empty((0, _nres, _desm), dtype = np.float32),
            'distance_matrix': np.empty((0, _nres, _nres), dtype = np.float32),
            'label': np.empty((0, 1), dtype = np.float32),
        }

        self.eval_data = copy.deepcopy(self.init_data) 
        self.start_idx = 0
        self.end_idx = 0
        self.data_it = 0
        self.batch_size = config.batch_size
        self.n_device = jax.local_device_count()

        """
            Define tool functions.
        """
        self._forward_and_backward = jax.value_and_grad(self._forward, has_aux=True)
        self._jit_forward = jax.jit(self._forward)
        self._jit_train_one_step = jax.jit(self._train_one_step)
        self._jit_eval_net_apply = jax.jit(self.eval_net.apply)
        self._jit_train_one_step_pmap = jax.pmap(jax.jit(self._train_one_step_pmap), axis_name='i')
        self._jit_eval_net_apply_pmap = jax.pmap(jax.jit(self.eval_net.apply), axis_name='i')
        self._tos_fn = self._jit_train_one_step if not self.pmap_flag else self._jit_train_one_step_pmap # train one step

    def update(self, idx_it):
        #### check for a new epoch
        if self.data_it >= len(self.name_list):
            self.data_it = 0
            np.random.shuffle(self.name_list)
        this_data_path = self.name_list[self.data_it]
        # self.recoder.info(f"--------------------------------------------------")
        # self.recoder.info(f"Loading data, from {this_data_path}...")
        self.data_it += 1
        #### update data
        with open(this_data_path, 'rb') as f:
            data_ = pkl.load(f)
        self.train_data = tree_map(
            lambda x, y: np.concatenate([x[idx_it - self.start_idx:], y], axis=0),
            self.train_data, data_,
        )

        self.start_idx = idx_it
        self.end_idx = idx_it + self.train_data['latent_embedding'].shape[0]
        # self.recoder.info(f"\tStart index now: {self.start_idx},")
        # self.recoder.info(f"\tEnd index now: {self.end_idx},")
        # self.recoder.info(f"--------------------------------------------------")
    
    def init_net(self, rng_key,):
        init_data = tree_map(lambda x: x[:1], self.init_data)
        init_data.pop('label')
        net_params = self.net.init(
            {'params': rng_key, 'dropout': rng_key},
            init_data,
            )
        return net_params
    
    def init_optimizer(self, net_params):
        return self.optimizer.init(net_params)
    
    def _forward(self, params, batch_data, rng_key):
        y = batch_data.pop('label') # (bs, 1)
        drop_key, rng_key = jax.random.split(rng_key)
        y_hat = self.net.apply(params, batch_data, rngs={'dropout': drop_key}) # (bs, 1)
        # mse loss
        assert y.ndim == y_hat.ndim
        loss = jnp.mean((jnp.square(y - y_hat)))
        return loss, (y_hat, rng_key)
    
    def _train_one_step(self, batch_data, params, opt_state, rng_key):
        loss, grad = self._forward_and_backward(params, batch_data, rng_key)
        loss, (pred, rng_key) = loss
        params_update, opt_state = self.optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, params_update)
        return loss, params, opt_state, rng_key
    
    def _train_one_step_pmap(self, batch_data, params, opt_state, rng_key):

        loss, grad = self._forward_and_backward(params, batch_data, rng_key)
        loss, (pred, rng_key) = loss
        loss = jax.lax.pmean(loss, axis_name='i')
        grad = pmean_tree(grad)
        params_update, opt_state = self.optimizer.update(grad, opt_state, params)
        params = optax.apply_updates(params, params_update)

        return loss, params, opt_state, rng_key
    
    def _shuffle_data(self, data,):
        """WARNING: better not to use before init data."""
        n_data = data['latent_embedding'].shape[0]
        indices = np.arange(n_data)
        np.random.shuffle(indices)
        shuffled_data = tree_map(lambda x: x[indices], data)
        return shuffled_data
    
    def check(self, step_it):

        start_it = step_it * self.batch_size
        stop_it = (step_it + 1) * self.batch_size 
        #### check if the data is enough
        if stop_it > self.end_idx:
            # breakpoint() ## check here
            self.update(start_it)
    
    def load_train_data(self, step_it):

        ### get indexes: (load_int * dbs * n_device)
        def _get_idxs():
            start_it = step_it * self.batch_size
            stop_it = (step_it + 1) * self.batch_size 
            return np.arange(start_it, stop_it) - self.start_idx

        _idx = _get_idxs()
        _data = tree_map(lambda x: x[_idx], self.train_data)
        if self.pmap_flag:
            _dbs = self.batch_size // self.n_device
            _data = tree_map(lambda x: np.reshape(x, (self.n_device, _dbs,) + x.shape[1:]), _data)
        return _data

    def train(self, rng_key, net_params, opt_state,):
        """main training function"""

        n_total_steps = self.config.n_total_steps
        best_eval_loss = self.eval(net_params, self.eval_data)
        best_params = net_params
        best_opt_state = opt_state
        n_save_steps = self.config.n_save_steps
        n_callback_steps = self.config.n_callback_steps
        os.makedirs(os.path.join(self.save_root, 'params'), exist_ok=True)
        os.makedirs(os.path.join(self.save_root, 'opt_states'), exist_ok=True)

        self.recoder.info(f'Start training, total steps = {n_total_steps}')
        start_time = datetime.datetime.now()
        t0 = datetime.datetime.now()

        # debug
        for step in range(n_total_steps):

            self.check(step)
            batch_data = self.load_train_data(step)
            loss, net_params, opt_state, rng_key = self._tos_fn(batch_data, net_params, opt_state, rng_key)
            
            
            if (step + 1) % n_callback_steps == 0:
                if self.pmap_flag: loss = loss[0]
                t1 = datetime.datetime.now()
                self.recoder.info(
                    f"Step {step + 1:<5d} | current loss: {loss:.4f} | time: {t1 - t0}")
                t0 = datetime.datetime.now()

            if (step + 1) % n_save_steps == 0:
                eval_loss = self.eval(net_params, self.eval_data)
                self.recoder.info(
                    f"Save Step {step + 1:<5d} | current eval loss: {eval_loss:.4f}")
                if eval_loss < best_eval_loss:
                    best_eval_loss, best_params, best_opt_state = eval_loss, net_params, opt_state
                
                if self.pmap_flag:
                    save_params, save_opt_state = tree_map(lambda arr: arr[0], (net_params, opt_state))
                else:
                    save_params, save_opt_state = net_params, opt_state
                save_path_ = os.path.join(self.save_root, f'params/params_step{step + 1}.pkl')
                with open(save_path_, 'wb') as f:
                    pkl.dump(tree_map(lambda arr: np.asarray(arr,), save_params), f)
                save_path_ = os.path.join(self.save_root, f'opt_states/opt_state_step{step + 1}.pkl')
                with open(save_path_, 'wb') as f:
                    pkl.dump(tree_map(lambda arr: np.asarray(arr,), save_opt_state), f)
            
        end_time = datetime.datetime.now()
        self.recoder.info(f"Training time: {end_time - start_time}, best eval loss: {best_eval_loss:.4f}.")
        
        # save best params and opt_state
        if self.pmap_flag:
            best_params, best_opt_state = tree_map(lambda arr: arr[0], (best_params, best_opt_state))
        else:
            best_params, best_opt_state = best_params, best_opt_state
        save_path_ = os.path.join(self.save_root, 'params', 'best_params.pkl')
        with open(save_path_, 'wb') as f:
            pkl.dump(tree_map(lambda arr: np.asarray(arr,), best_params), f)
        save_path_ = os.path.join(self.save_root, 'opt_states', 'best_opt_state.pkl')
        with open(save_path_, 'wb') as f:
            pkl.dump(tree_map(lambda arr: np.asarray(arr,), best_opt_state), f)
        return rng_key

    def eval(self, params, eval_data):
        bs = self.config.eval_batch_size
        n_device = jax.local_device_count()
        dbs = bs // n_device
        n_data = eval_data['latent_embedding'].shape[0]
        n_steps = int(np.ceil(n_data / bs))
        pred_labels = []
        for step in range(n_steps):
            start_idx = step * bs
            stop_idx = start_idx + bs
            batch_inputs = tree_map(lambda x: x[np.arange(start_idx, stop_idx) % n_data], eval_data)
            if self.pmap_flag:
                batch_inputs = tree_map(lambda x: np.reshape(x, (n_device, dbs,) + x.shape[1:]), batch_inputs)
                pred_ = self._jit_eval_net_apply_pmap(params, batch_inputs,) # (bs, out)
            else:
                pred_ = self._jit_eval_net_apply(params, batch_inputs,)
            pred_ = np.asarray(pred_, dtype=np.float32).reshape(-1, pred_.shape[-1]) # (bs, out)
            pred_labels.append(pred_) 
        pred_labels = np.concatenate(pred_labels, axis=0)[:n_data] # (n_data, out)
        eval_loss = np.mean((pred_labels - eval_data['label']) ** 2)
        return eval_loss

def set_train_config(
        n_total_steps: int,
        batch_size: int,
        eval_batch_size: int,
        split_rate: float = 0.9,
        n_save_steps: int = 1000,
        n_callback_steps: int = 100,
        max_learning_rate: float = 2e-4,
        weight_decay: float = 1e-4,
    ):
    train_config = {
        'n_total_steps': n_total_steps,
        'batch_size': batch_size,
        'eval_batch_size': eval_batch_size,
        'split_rate': split_rate,
        'n_save_steps': n_save_steps,
        'n_callback_steps': n_callback_steps,
        'learning_rate': {
            'min': min(1e-5, max_learning_rate / 10),
            'max': max_learning_rate,
            'warmup_steps': min(1000, n_total_steps // 10),
            'decay_steps': n_total_steps,
        },
        'weight_decay': weight_decay,
    }
    return ConfigDict(train_config)
