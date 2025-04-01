"""
    Train dataloader.
"""
import jax
import time
import numpy as np
import pickle as pkl
import concurrent.futures

from jax.tree_util import tree_map

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
            data_to_update = pkl.load(f)['latent']
        self.data = np.concatenate(
            [self.data[idx_it - self.start_idx:], data_to_update], axis = 0) ### (N, T, D)
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
        return self.data[_idx]
    
    def load_data(self, step_it,):

        batch_data = self.organize(step_it)
        batch_data = np.reshape(
            batch_data, (self.num_local_devices, self.device_batch_size,) + batch_data.shape[1:])
        # breakpoint()
        batch_data *= np.sqrt(self.data_config['latent_dim'])
        return {'feat': batch_data}
    
    def load_init_data(self,):

        return {'feat': self.data[:1]}