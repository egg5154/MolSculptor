"""
    Train dataloader.
"""
import jax
import time
import copy
import random
import numpy as np
import jax.numpy as jnp
import pickle as pkl
import concurrent.futures

from jax.tree_util import tree_map
from typing import Optional
from functools import partial
from rdkit.Chem import DataStructs
from jax.experimental.multihost_utils import process_allgather
from dataclasses import dataclass
from .utils import print_nested_dict

def read_file(file_path):
    with open(file_path, 'rb') as f:
        return pkl.load(f)
    
def read_files_in_parallel(file_paths, num_parallel_worker = 32):

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_parallel_worker) as executor:
        results = list(executor.map(read_file, file_paths))

    return results

class TrainDataLoader:

    def __init__(self, name_list, recoder, data_config,):

        self.name_list = name_list ### a list of dirs
        self.data_config = data_config
        self.process_id = jax.process_index()
        # self.load_int = data_config['pre_load_int']
        self.data_it = 0
        self.recoder = recoder

        ### create indexes and shuffle
        np.random.seed(data_config['seed'])

        self.device_batch_size = data_config['batch_size_device']
        self.num_local_devices = jax.local_device_count()
        self.gbs = self.device_batch_size * jax.device_count()
        self.lbs = self.device_batch_size * jax.local_device_count()
        self.data = np.empty(
            (0, data_config['n_query_tokens'], data_config['latent_dim']), dtype = np.float32)
        self.smiles = np.empty((0,), dtype = object)
        self.labels = {
            'qed': np.empty((0,), np.float32),
            'sa': np.empty((0,), np.float32),
            'logp': np.empty((0,), np.float32),
            'mw': np.empty((0,), np.float32),
        }
        self.start_idx = 0
        self.end_idx = 0

    def update(self, idx_it):
        #### check for a new epoch
        if self.data_it >= len(self.name_list):
            self.data_it = 0
            np.random.shuffle(self.name_list)
        this_data_path = self.name_list[self.data_it]
        self.recoder.info(f"--------------------------------------------------")
        self.recoder.info(f"Loading data, from {this_data_path}...")
        self.data_it += 1
        #### update data
        with open(this_data_path, 'rb') as f:
            x_ = pkl.load(f)
            data_to_update = x_['latent']
            smiles_to_update = x_['smiles']
            labels_to_update = {k: x_[k] for k in self.labels.keys()}
        self.data = np.concatenate(
            [self.data[idx_it - self.start_idx:], data_to_update], axis = 0) # (N, T, D)
        self.smiles = np.concatenate(
            [self.smiles[idx_it - self.start_idx:], smiles_to_update], axis = 0) # (N,)
        self.labels = {
            k: np.concatenate(
                [self.labels[k][idx_it - self.start_idx:], labels_to_update[k]], axis = 0,
                ) for k in self.labels.keys()
        }
        self.start_idx = idx_it
        self.end_idx = idx_it + self.data.shape[0]
        self.recoder.info(f"\tStart index now: {self.start_idx},")
        self.recoder.info(f"\tEnd index now: {self.end_idx},")
        self.recoder.info(f"--------------------------------------------------")
        # breakpoint() ## check here
    
    def check(self, step_it):

        start_it = step_it * self.gbs + self.lbs * self.process_id
        stop_it = step_it * self.gbs + self.lbs * (self.process_id + 1)
        #### check if the data is enough
        if stop_it > self.end_idx:
            # breakpoint() ## check here
            self.update(start_it)

    def shuffle_indexes(self,):
        np.random.shuffle(self.name_list)

    def organize(self, step_it):

        ### get indexes: (load_int * dbs * n_device)
        def _get_idxs():
            start_it = step_it * self.gbs + self.lbs * self.process_id
            stop_it = step_it * self.gbs + self.lbs * (self.process_id + 1)
            return np.arange(start_it, stop_it) - self.start_idx

        _idx = _get_idxs()
        # breakpoint() ## check here
        return self.data[_idx], self.smiles[_idx], tree_map(lambda x: x[_idx], self.labels)
    
    def load_data(self, step_it,):

        batch_data, batch_smiles, batch_labels = self.organize(step_it)
        batch_data = np.reshape(
            batch_data, (self.num_local_devices, self.device_batch_size,) + batch_data.shape[1:])
        batch_data *= np.sqrt(self.data_config['latent_dim'])

        ### deprecated: just-in-time calculation
        # def _calculate_all_properties(smi):
        #     mol = Chem.MolFromSmiles(smi)
        #     if mol is None:
        #         return (np.nan, np.nan, np.nan, np.nan)
            
        #     try:
        #         qed = QED.qed(mol)
        #         sa = sascorer.calculateScore(mol)
        #         logp = Crippen.MolLogP(mol)
        #         mw = Descriptors.MolWt(mol)
        #         return (qed, sa, logp, mw)
        #     except Exception:
        #         return (np.nan, np.nan, np.nan, np.nan)
        # batch_smiles = np.reshape(
        #     batch_smiles, (self.num_local_devices, self.device_batch_size,)
        # )
        # batch_mols = [Chem.MolFromSmiles(s) for s in batch_smiles]
        # batch_labels = {
        #     # 'qed': np.array([QED.qed(mol) for mol in batch_mols], dtype = np.float32),
        #     # 'sa': np.array([sascorer.calculateScore(mol) for mol in batch_mols], dtype = np.float32),
        #     'qed': np.array(Parallel(n_jobs=-1)(delayed(QED.qed)(mol) for mol in batch_mols), dtype = np.float32),
        #     'sa': np.array(Parallel(n_jobs=-1)(delayed(sascorer.calculateScore)(mol) for mol in batch_mols), dtype = np.float32),
        #     'logp': np.array([Crippen.MolLogP(mol) for mol in batch_mols], dtype = np.float32),
        #     'mw': np.array([Descriptors.MolWt(mol) for mol in batch_mols], dtype = np.float32),
        # }
        # all_properties_list = Parallel(n_jobs=-1)(
        #     delayed(_calculate_all_properties)(smi) for smi in batch_smiles
        # )
        # qed_list, sa_list, logp_list, mw_list = zip(*all_properties_list)
        # batch_labels = {
        #     'qed': np.array(qed_list, dtype=np.float32),
        #     'sa': np.array(sa_list, dtype=np.float32),
        #     'logp': np.array(logp_list, dtype=np.float32),
        #     'mw': np.array(mw_list, dtype=np.float32),
        # }

        ### load from data
        batch_labels = tree_map(
            lambda x: np.reshape(x, (self.num_local_devices, self.device_batch_size,)),
            batch_labels,
        )
        force_drop_ids = np.random.binomial(1, self.data_config.force_drop_rate, batch_smiles.shape).astype(np.int16)
        force_drop_ids = np.reshape(
            force_drop_ids, (self.num_local_devices, self.device_batch_size,)
        )

        return {'feat': batch_data, 'labels': batch_labels, 'force_drop_ids': force_drop_ids}
    
    def load_init_data(self,):

        return {
            'feat': self.data[:1], 
            'labels': tree_map(lambda x: x[:1], self.labels), 
            'force_drop_ids': np.zeros((1,), dtype=np.int16)
            }

### for ae training
DTYPE = np.int16
def random_choice_fn(array, p, size):

    random_value = np.random.uniform(0, 1, size)
    sum_p = np.cumsum(p)
    index = np.searchsorted(sum_p, random_value)
    result = array[index]

    return result

def recover_from_csr(bond_feature):

    keys = ['bond_type', 'bond_mask', 'stereo', 'conjugated', 'in_ring', 'graph_distance']
    for k in keys:
        bond_feature[k] = bond_feature[k].toarray()
    
    return bond_feature

def make_graph_feature(graph_feature,
                       n_padding_atom: int = 32,
                       dtype: np.dtype = np.int16):

    key_dict = {
        'atom_features': list(graph_feature[0]['atom_features'].keys()), 
        'bond_features': list(graph_feature[0]['bond_features'].keys()),
    }
    
    batched_graph_feature = {
        top_key: {sub_key: 
            np.stack([this_graph[top_key][sub_key] for this_graph in graph_feature]) for sub_key in key_dict[top_key]
        } for top_key in key_dict.keys()
    }

    return batched_graph_feature

def samp_dict_fn(dic: dict):
    idxes = []
    for v in dic.values():
        idxes.append(random.choice(v))
    return np.asarray(idxes, np.int32)

def make_sequence_feature(seq_tokens, data_config,):

    n_pad_to_tokens = data_config['n_pad_token']
    n_prefix = data_config['n_query_tokens']
    eos_token = data_config['eos_token_id']
    mask_token = data_config['mask_token_id']
    k = data_config['n_repeat_k']
    sampling_method = data_config['sampling_method']
    assert sampling_method in ['random', 'clustered'] ### updated 11-13: support clustered

    def padding_prefix(arr, n_pad_to_tokens = n_pad_to_tokens, 
                       n_prefix = n_prefix, mask_token = mask_token):
        ### process arr label for prefix tokens
        ### arr shape: (n, n_repeats, n_tokens)
        input_tokens = arr['input_tokens']
        input_mask = arr['input_mask']
        label = arr['label']
        label_mask = arr['label_mask']
        ### check
        assert np.ndim(input_tokens) == 3, "Input array tokens' dim should be 3!"

        ### padding for prefix tokens
        label = np.pad(label, ((0, 0), (0, 0), (n_prefix, 0)), mode='constant', constant_values=mask_token)
        label_mask = np.pad(label_mask, ((0, 0), (0, 0), (n_prefix, 0)), mode='constant', constant_values=0)
        label = label[:, :, :n_pad_to_tokens]
        label_mask = label_mask[:, :, :n_pad_to_tokens]
        processed_arr = {
            'input_tokens': input_tokens,
            'input_mask': input_mask,
            'label': label,
            'label_mask': label_mask,
        }
        return processed_arr

    def pad_or_crop_k(arr_dict, k = k,):

        input_tokens = arr_dict['input_tokens']
        input_mask = arr_dict['input_mask']
        label = arr_dict['label']
        label_mask = arr_dict['label_mask']

        ## for random smiles (deprecated)
        n_sf = input_tokens.shape[0]
        if n_sf <= k: ## If n_sf <= k, just padding
            n_pad = k - n_sf
            input_tokens = np.pad(input_tokens, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)
            input_mask = np.pad(input_mask, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)
            label = np.pad(label, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)
            label_mask = np.pad(label_mask, ((0, n_pad), (0, 0)), mode='constant', constant_values=0)
        else: ## n_sf > k

            ### updated 11-13
            if sampling_method == 'random':
                cano_index = max(arr_dict['cano_index'], 0)
                choose_idxs = np.delete(np.arange(n_sf, dtype=np.int16), cano_index)
                # shuffle_idxs = np.random.choice(choose_idxs, size=k-1, replace=False)
                np.random.shuffle(choose_idxs)
                # breakpoint() ### check
                shuffle_idxs = choose_idxs[:k-1]
                shuffle_idxs = np.insert(shuffle_idxs, 0, cano_index)

                input_tokens = input_tokens[shuffle_idxs]
                input_mask = input_mask[shuffle_idxs]
                label = label[shuffle_idxs]
                label_mask = label_mask[shuffle_idxs]
            else: ### clustered
                cluster_dict = arr_dict['cluster_dict'] # (n_sf,)
                shuffle_idxs = samp_dict_fn(cluster_dict)

                input_tokens = input_tokens[shuffle_idxs]
                input_mask = input_mask[shuffle_idxs]
                label = label[shuffle_idxs]
                label_mask = label_mask[shuffle_idxs]

        return {
            'input_tokens': input_tokens, 
            'input_mask': input_mask, 
            'label': label, 
            'label_mask': label_mask,
        } ### (n_repeat_k, n_tokens)

    pad_or_crop_k_fn = partial(pad_or_crop_k, k = k)
    seq_tokens = [pad_or_crop_k_fn(x) for x in seq_tokens]
    train_dict = {
        'input_tokens': np.stack([data['input_tokens'] for data in seq_tokens], axis=0), ### (num_batches*batch_size, num_repeats, num_tokens)
        'input_mask': np.stack([data['input_mask'] for data in seq_tokens], axis=0),
        'label': np.stack([data['label'] for data in seq_tokens], axis=0),
        'label_mask': np.stack([data['label_mask'] for data in seq_tokens], axis=0),
    }
    train_dict = padding_prefix(train_dict) ### (num_batches*batch_size, num_repeats, num_tokens)

    return train_dict

def make_feature(data_list,
                 data_config,):

    graph_feature = [data['graph_features'] for data in data_list]
    seq_tokens = [data['tokens'] for data in data_list]
    # for g in graph_feature:
    #     g['bond_features'] = recover_from_csr(g['bond_features'])

    graph_dict = make_graph_feature(graph_feature, data_config['n_pad_atom'], data_config['dtype'])
    seq_dict = make_sequence_feature(seq_tokens, data_config,)

    return graph_dict, seq_dict

def organize_name_list(train_name_list, selected_index):
    ### name list: dict with keys 'path' and 'id'
    organized_name_list = train_name_list['path'][selected_index]
    return organized_name_list

class AEDataLoader:

    def __init__(self, name_list, data_config,):

        self.name_list = name_list
        self.data_config = data_config
        self.process_id = jax.process_index()
        self.load_int = data_config['pre_load_int']
        self.device_partition = data_config['device_partition']
        assert self.device_partition.sum() == jax.device_count()

        ### create indexes and shuffle
        self.num_data = len(self.name_list['path'])
        self.data_indexes = np.arange(self.num_data)
        np.random.seed(data_config['seed'])
        np.random.shuffle(self.data_indexes)

        self.device_batch_size = data_config['batch_size_device']
        self.num_local_devices = jax.local_device_count()
        self.global_batch_size = self.device_batch_size * jax.device_count()
        self.local_batch_size = self.device_batch_size * jax.local_device_count()

    def shuffle_indexes(self,):

        np.random.shuffle(self.data_indexes)

    def organize(self, step_it):

        ### get indexes: (load_int * dbs * n_device)
        def _get_idxs(this_size,):
            start_it = step_it * self.global_batch_size + self.local_batch_size * self.load_int * self.process_id
            stop_it = step_it * self.global_batch_size + self.local_batch_size * self.load_int * (self.process_id + 1)
            return np.int64(np.arange(start_it, stop_it) % this_size)

        load_idx = _get_idxs(self.num_data,)
        return organize_name_list(self.name_list, load_idx,) ## (num_batches * n_local_devices * dbs)
    
    def _make_feature(self, name_list_trunk, data_config):

        #### make batched data
        data_from_pickle = read_files_in_parallel(name_list_trunk, data_config['n_workers'])
        graph_feat, sequence_feat = make_feature(data_from_pickle, data_config,)
        input_feat = {
            'graph_features': graph_feat,
            'sequence_features': {'tokens': np.array(sequence_feat['input_tokens'], data_config['dtype']), 
                                  'mask': np.array(sequence_feat['input_mask'], data_config['dtype']),}
        }
        label = {
            'label': np.array(sequence_feat['label'], data_config['dtype']),
            'mask': np.array(sequence_feat['label_mask'], data_config['dtype']),
        }
        return input_feat, label

    def load_data(self, step_it,):

        name_list_trunk = self.organize(step_it)
        data_config = self.data_config
        input_feat, label = self._make_feature(name_list_trunk, data_config)

        return input_feat, label

    def load_init_data(self,):

        init_name_list = self.name_list['path'][:self.device_batch_size]
        data_config = copy.deepcopy(self.data_config)
        data_config['pre_load_int'] = 1
        init_data = self._make_feature(init_name_list, data_config)
        print("Data shape and dtypes:")
        for _data in init_data:
            print_nested_dict(
                tree_map(lambda arr: (arr.shape, arr.dtype), _data), prefix = "\t"
            )
        return init_data
    
    def select_data(self, data, data_it):

        pid = self.process_id
        local_batch_size = self.local_batch_size
        start_idx = data_it * local_batch_size
        stop_idx = (data_it + 1) * local_batch_size

        data_it = tree_map(lambda arr: arr[start_idx: stop_idx], data)
        data_it = tree_map(lambda arr: jnp.reshape(arr, (self.num_local_devices, self.device_batch_size,) + arr.shape[1:]), data_it)

        return data_it