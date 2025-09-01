"""main function for training 1d stable moledit model."""

import os
import sys
import jax
import copy
import math
import logging
import jax.numpy as jnp
import pickle as pkl
import numpy as np
import argparse

def arg_parse():

    parser = argparse.ArgumentParser(description='Inputs for main.py')
    parser.add_argument('--num_total_epochs', type=int, default=100, help='Number of total epochs.')
    parser.add_argument('--name_list_path', type=str)
    parser.add_argument('--gradient_clip', type=float, default=1.0, help='gradient clip value')
    parser.add_argument('--logger_path', type=str, default="./logs/log_train_vq.txt", help='Location of logger file.')
    parser.add_argument('--params_path', type=str, help='Location of param file.')
    parser.add_argument('--opt_state_path', type=str, help='Location of optimizer state file.')
    parser.add_argument('--config_path', type=str, help='Location of config file.')
    parser.add_argument('--save_ckpt_path', type=str, default="./protoken_params/train_vq", help='Location of checkpoint file.')
    parser.add_argument('--random_seed', type=int, default=8888, help="random seed")
    parser.add_argument('--np_random_seed', type=int, default=8888, help="np random seed")
    parser.add_argument('--start_step', type=int, default=0, help="start step")
    parser.add_argument('--callback_steps', type=int, default=10000, help="callback steps")
    parser.add_argument('--save_steps', type=int, default=10000, help="save ckpt steps")
    parser.add_argument('--pre_load_steps', type=int, default=10, help="pre load steps")
    parser.add_argument('--device_batch_size', type=int, default=8, help="device batch size")
    # parser.add_argument('--device_partition', nargs='+', type=int, default=None, help='device partition for datasets')
    
    ## distributed
    parser.add_argument('--coordinator_address', type=str, help="coordinator address")
    parser.add_argument('--num_processes', type=int, default=2, help="number of processes")
    parser.add_argument('--device_ids', nargs='+', type=int, default=None, help='local device ids')
    parser.add_argument('--rank', type=int, default=0, help="rank")

    arguments = parser.parse_args()
    
    return arguments

args = arg_parse()

##### Initializing distributed
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".95"
jax.distributed.initialize(
    coordinator_address = args.coordinator_address,
    num_processes = args.num_processes,
    process_id = args.rank,
    local_device_ids = args.device_ids or [0, 1, 2, 3, 4, 5, 6, 7],
)
RANK = jax.process_index()
print("This rank:", RANK)

import optax
import datetime

from jax import value_and_grad, jit, vmap
from jax.tree_util import tree_map
from functools import partial
from flax import linen as nn
from tqdm import tqdm
from jax.sharding import PositionalSharding
from jax.debug import breakpoint
from jax.experimental.multihost_utils import process_allgather
from flax.jax_utils import replicate
from ml_collections.config_dict import ConfigDict

sys.path.append(os.path.dirname(__file__))
from net.generator import L2SeqGenerator as InfoVAESeqGenerator
from train.withloss import L2SeqGenWithLoss as InfoSeqGenWithLoss
from train.utils import split_multiple_rng_keys, psum_tree, pmean_tree, print_net_params, print_net_params_count
from train.dataloader import AEDataLoader

def load_ckpt(path):
    with open(path, 'rb') as f:
        params = pkl.load(f)
        params = tree_map(jnp.asarray, params)
    return params

def save_ckpt(path, params):
    ## save as numpy array
    with open(path, 'wb') as f:
        pkl.dump(tree_map(np.array, params), f)

def train():

    #### set recoder
    recoder = logging.getLogger("training stable moledit")
    recoder.setLevel(level=logging.DEBUG)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level=logging.DEBUG) 
    recoder.addHandler(stream_handler)
    if (RANK == 0): # add file handler for only rank 0
        file_handler = logging.FileHandler(args.logger_path)
        file_handler.setLevel(level=logging.DEBUG)
        recoder.addHandler(file_handler)
    
    #### prepare name list
    print("Loading name list...")
    with open(args.name_list_path, 'rb') as f:
        name_list = pkl.load(f)
    # load config from config path
    with open(args.config_path, 'rb') as f:
        config_dicts = pkl.load(f)
    global_config = ConfigDict(config_dicts['global_config'])
    net_config = ConfigDict(config_dicts['net_config'])
    train_config = ConfigDict(config_dicts['train_config'])
    data_config = config_dicts['data_config']      

    #### set constants
    DEVICE_BATCH_SIZE = args.device_batch_size # batch size per device
    N_GLOBAL_DEVICES = jax.device_count()
    N_LOCAL_DEVICES = jax.local_device_count()
    N_EPOCHS = args.num_total_epochs
    N_DATAS = name_list['path'].size
    N_STEPS_PER_EPOCH = N_DATAS // (N_GLOBAL_DEVICES * DEVICE_BATCH_SIZE)
    TOT_STEPS = N_STEPS_PER_EPOCH * N_EPOCHS

    #### save config
    data_config['batch_size_device'] = DEVICE_BATCH_SIZE
    data_config['pre_load_int'] = args.pre_load_steps
    data_config['seed'] = args.np_random_seed
    with open(os.path.join(args.save_ckpt_path, f"config.pkl"), 'wb') as f:
        pkl.dump({
            'global_config': global_config.to_dict(),
            'net_config': net_config.to_dict(),
            'train_config': train_config.to_dict(),
            'data_config': data_config,
        }, f)

    #### print basic info
    recoder.info("DATA INFO:")
    recoder.info("\tNDATAS: {}".format(N_DATAS))
    recoder.info("\tSTEPS_PER_EPOCH: {}".format(N_STEPS_PER_EPOCH))
    args_dict = vars(args)
    recoder.info("INPUT ARGS:")
    for k, v in args_dict.items():
        recoder.info(f"\t{k}: {v}")

    #### initialize network, optimizer and dataloader
    generator = InfoVAESeqGenerator(net_config, global_config)
    withloss_net = InfoSeqGenWithLoss(train_config, global_config, generator)
    init_net = InfoSeqGenWithLoss(train_config, global_config, generator, False)
    lr_config = train_config.learning_rate
    learning_rate_schedule = optax.warmup_cosine_decay_schedule(
        init_value = lr_config.min,
        peak_value = lr_config.max,
        warmup_steps = lr_config.warmup_steps,
        decay_steps = lr_config.decay_steps, 
        end_value = lr_config.min,
    )
    # optimizer = optax.chain(optax.clip_by_global_norm(args.gradient_clip),
    #                         optax.adam(learning_rate = learning_rate_schedule))
    optimizer = optax.adamw(
        learning_rate = learning_rate_schedule,
        weight_decay = train_config.weight_decay,
    )
    dataloader = AEDataLoader(name_list, data_config)

    #### load/init params & opt state
    def _init_params(rng_key):

        init_data = dataloader.load_init_data()
        state_key, param_key, dropout_key, rng_key = jax.random.split(rng_key, 4)
        init_keys = {
            'latent': state_key, 'params': param_key, 'dropout': dropout_key,
        }
        params = init_net.init(init_keys, *init_data)
        params = tree_map(np.asarray, params) ## release memory
        return params, rng_key

    rng_key = jax.random.PRNGKey(args.random_seed)
    np.random.seed(args.np_random_seed)
    if args.params_path:
        params = load_ckpt(args.params_path) ## jax array
    else:
        params, rng_key = _init_params(rng_key)
        save_path = os.path.join(
            args.save_ckpt_path, f"params_step0.pkl"
        )
        save_ckpt(save_path, params)
    
    if args.opt_state_path:
        opt_state = load_ckpt(args.opt_state_path)
    else:
        opt_state = optimizer.init(params)
        save_path = os.path.join(
            args.save_ckpt_path, f"opt_state_step0.pkl"
        )
        save_ckpt(save_path, opt_state)
    
    #### print net params info
    encoder_n_params = print_net_params_count(params['params']['generator']['Encoder_0'])
    recoder.info(f"Encoder params count: {encoder_n_params}")
    decoder_n_params = print_net_params_count(params['params']['generator']['Decoder_0'])
    recoder.info(f"Decoder params count: {decoder_n_params}")
    decoder_n_params = print_net_params_count(params['params']['generator']['Dense_0'])
    recoder.info(f"Dense params count: {decoder_n_params}")
    decoder_n_params = print_net_params_count(params['params']['generator']['Projector_0'])
    recoder.info(f"Projector params count: {decoder_n_params}")
    # breakpoint()
    params = replicate(params)
    opt_state = replicate(opt_state)

    #### define functions
    def forward(net_params, batch_data, step_it, rng_keys):
        loss, loss_dict = withloss_net.apply(
            net_params, *batch_data, step_it, rngs = rng_keys,
        )
        return loss, loss_dict
    forward_and_backward = jax.value_and_grad(forward, has_aux=True)
    
    def train_one_step(batch_data, train_state):
        net_params, opt_state, rng_key, step_it = \
            train_state['params'], train_state['opt_state'], train_state['rng_key'], train_state['step_it']
        
        dropout_key, rng_key = jax.random.split(rng_key)
        latent_key, rng_key = jax.random.split(rng_key)
        input_rng_key = {'dropout': dropout_key, 'latent': latent_key}
        loss_dict, grad_dict = forward_and_backward(
            net_params, batch_data, step_it, input_rng_key)
        loss, loss_dict = loss_dict
        loss_dict = pmean_tree(loss_dict)
        grad_dict = pmean_tree(grad_dict)
        params_update, opt_state = jax.jit(optimizer.update)(grad_dict, opt_state, net_params)
        # #### debug
        # breakpoint()
        net_params = jax.jit(optax.apply_updates)(net_params, params_update)

        train_state['params'] = net_params
        train_state['opt_state'] = opt_state
        train_state['rng_key'] = rng_key
        train_state['step_it'] += 1
    
        return loss_dict, train_state
    train_one_step_pmap = jax.pmap(jax.jit(train_one_step), axis_name='i', donate_argnums=(1,))
        
    ################## training ##################
    train_state = {
        'params': params,
        'opt_state': opt_state,
        'rng_key': jax.random.split(rng_key, N_LOCAL_DEVICES,),
        'step_it': np.zeros((N_LOCAL_DEVICES,), dtype=np.int16) + args.start_step,
    }
    if (RANK == 0):
        recoder.info("=====================START TRAINING=====================")
    pre_load_steps = args.pre_load_steps

    for step in range(args.start_step, TOT_STEPS):

        if (step % args.callback_steps == 0 or step == args.start_step):
            start_time = datetime.datetime.now()
        
        ## pre load data
        if (step % pre_load_steps == 0 or step == args.start_step):
            start_time_load = datetime.datetime.now()
            
            pre_load_data_dict = dataloader.load_data(step)
            #### data sink mode
            # pre_load_data_dict = tree_map(lambda arr: jnp.asarray(arr), pre_load_data_dict)
            end_time_load = datetime.datetime.now()
            data_it = 0
            pre_load_time = end_time_load - start_time_load

        ## select data
        data_dict = dataloader.select_data(pre_load_data_dict, data_it)
        # #### debug
        data_it += 1

        ## training
        loss_dict, train_state = train_one_step_pmap(data_dict, train_state)
        # breakpoint()
        ## callback
        if (RANK == 0) and ((step + 1) % args.callback_steps == 0):

            end_time = datetime.datetime.now()

            recoder.info(f"[Callback] Step {(step + 1)}, info:")
            for k, v in loss_dict.items():
                recoder.info(f"\t{k}: {v[0]:.4f}")
            recoder.info(f"\tTime: {end_time - start_time} & {pre_load_time}")
        
        if (RANK == 0) and ((step + 1) % args.save_steps == 0):
            # assert False ## debug
            
            save_path = os.path.join(
                args.save_ckpt_path, f"params/params_step{(step + 1)}.pkl"
            )
            save_params = tree_map(lambda arr: arr[0], train_state['params'])
            save_ckpt(save_path, save_params)

            save_path = os.path.join(
                args.save_ckpt_path, f"opt_states/opt_state_step{(step + 1)}.pkl"
            )
            save_opt_state = tree_map(lambda arr: arr[0], train_state['opt_state'])
            save_ckpt(save_path, save_opt_state)

    if (RANK == 0):
        recoder.info(f"=====================END TRAINING=====================")        
        f.close()
    print("done")

if __name__ == '__main__':

    train()
    jax.distributed.shutdown()