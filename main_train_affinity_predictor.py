
import logging
import jax.numpy as jnp
import pickle as pkl
import numpy as np
import argparse
import os
import jax
import optax

import argparse
def arg_parse():

    parser = argparse.ArgumentParser(description="Train affinity predictor")
    parser.add_argument('--config_path', type=str, help='Path to config file')
    # parser.add_argument('--data_root', type=str, required=True, help='Root directory of the data')
    parser.add_argument('--save_ckpt_path', type=str, required=True, help='Directory to save checkpoints')
    parser.add_argument('--save_logger_path', type=str, required=True, help='Path to save logs')
    parser.add_argument('--load_params_path', type=str, help='Path to initial parameters')
    parser.add_argument('--load_opt_state_path', type=str, help='Path to optimizer state')
    parser.add_argument('--load_split_indices_path', type=str, help='Path to split indices')
    parser.add_argument('--n_total_steps', type=int, required=True, help='Total training steps')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size for training')
    parser.add_argument('--callback_steps', type=int, required=True, help='Steps between saving checkpoints')
    parser.add_argument('--save_steps', type=int, required=True, help='Steps between saving logs')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--name_list_path', type=str, required=True, help='Path to the list of data names')

    args = parser.parse_args()
    return args

args = arg_parse()

from flax.jax_utils import replicate
from jax.tree_util import tree_map, tree_map_with_path, DictKey
from utils import set_recoder
from ml_collections import ConfigDict
from net.affinity_predictor import set_projector_config, set_encoder_config, \
    set_global_config, PretrainPredictorWithPocket
from train.train_affinity_predictor import Trainer, set_train_config
from train.utils import print_net_params_count

def load_ckpt(path):
    with open(path, 'rb') as f:
        params = pkl.load(f)
        params = tree_map(jnp.asarray, params)
    return params

def save_ckpt(path, params):
    ## save as numpy array
    with open(path, 'wb') as f:
        pkl.dump(tree_map(np.array, params), f)

HIDDEN_SIZE = 16
N_HEAD = HIDDEN_SIZE // 4
N_POCKET_LAYERS = 3
N_FUSION_LAYERS = 3
SHARED_WEIGHTS = True
DROPOUT_RATE = 0.1

def main(args):

    recoder = set_recoder(args.save_logger_path)
    with open(args.name_list_path, 'rb') as f:
        name_list = pkl.load(f)
    """
        Load config
    """
    if args.config_path:
        with open(args.config_path, 'rb') as f:
            config_dicts = pkl.load(f)
        global_config = ConfigDict(config_dicts['global_config'])
        encoder_config = ConfigDict(config_dicts['encoder_config'])
        projector_config = ConfigDict(config_dicts['projector_config'])
        train_config = ConfigDict(config_dicts['train_config'])
    else:
        global_config = set_global_config()
        encoder_config = set_encoder_config(
            shared_weights = SHARED_WEIGHTS,
            hidden_size = HIDDEN_SIZE,
            n_head = N_HEAD,
            n_pocket_encoder_layers = N_POCKET_LAYERS,
            n_fusion_encoder_layers = N_FUSION_LAYERS,
            dropout_rate = DROPOUT_RATE,
        )
        projector_config = set_projector_config(
            input_dim = HIDDEN_SIZE,
            scale = 2.5,
            shift = 9.0,
            dropout_rate = DROPOUT_RATE,
        )
        train_config = set_train_config(
            n_total_steps = args.n_total_steps,
            batch_size = args.batch_size,
            eval_batch_size = args.batch_size,
            split_rate = 0.9,
            n_save_steps = args.save_steps,
            n_callback_steps = args.callback_steps,
            weight_decay = 1e-3,
        )
        # save config
        config_dicts = {
            'global_config': global_config.to_dict(),
            'encoder_config': encoder_config.to_dict(),
            'projector_config': projector_config.to_dict(),
            'train_config': train_config.to_dict(),
        }
        with open(os.path.join(args.save_ckpt_path, f"config.pkl"), 'wb') as f:
            pkl.dump(config_dicts, f)
    
    """
        Print info
    """
    args_dict = vars(args)
    recoder.info("INPUT ARGS:")
    for k, v in args_dict.items():
        recoder.info(f"\t{k}: {v}")
    
    """
        Set network & optimizer
    """
    net = PretrainPredictorWithPocket(
        global_config=global_config,
        encoder_config=encoder_config,
        projector_config=projector_config,
    )
    optimizer = optax.adamw(
        learning_rate = optax.warmup_cosine_decay_schedule(
            init_value = train_config.learning_rate.min,
            peak_value = train_config.learning_rate.max,
            warmup_steps = train_config.learning_rate.warmup_steps,
            decay_steps = train_config.learning_rate.decay_steps,
            end_value = train_config.learning_rate.min,
        ),
        weight_decay = train_config.weight_decay,
    )
    trainer = Trainer(
        config=train_config,
        net=net,
        recoder=recoder,
        optimizer=optimizer,
        # data_root=args.data_root,
        name_list=name_list,
        save_root=args.save_ckpt_path,
        pmap_flag=False,
    )

    rng_key = jax.random.PRNGKey(args.random_seed)
    np.random.seed(args.random_seed)
    """
        Init net / optimizer / data
    """
    split_indices = None
    if args.load_split_indices_path:
        split_indices = np.load(args.load_split_indices_path,)
    # trainer.split_data(trainer.full_data, indices=split_indices,)
    if args.load_params_path:
        net_params = load_ckpt(args.load_params_path)
    else:
        net_params = trainer.init_net(rng_key)
        save_path = os.path.join(
            args.save_ckpt_path, f"params_step0.pkl"
        )
        save_ckpt(save_path, net_params)
    if args.load_opt_state_path:
        opt_state = load_ckpt(args.load_opt_state_path)
    else:
        opt_state = trainer.init_optimizer(net_params,)
        save_path = os.path.join(
            args.save_ckpt_path, f"opt_state_step0.pkl"
        )
        save_ckpt(save_path, opt_state)
    
    # print net params info
    n_params = print_net_params_count(net_params['params'])
    recoder.info(f"Params count: {n_params}")
    breakpoint()
    if trainer.pmap_flag:
        net_params = replicate(net_params)
        opt_state = replicate(opt_state)
        rng_key = jax.random.split(rng_key, jax.local_device_count())

    """
        Start training"""
    rng_key = trainer.train(rng_key, net_params, opt_state)

    print('done')

if __name__ == "__main__":
    main(args)
