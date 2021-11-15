import torch
import torch.utils.data as tordata
import torch.distributed as dist
import math
import random
import numpy as np

class TripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size):
        self.dataset = dataset
        self.batch_size = batch_size

    def __iter__(self):
        while (True):
            sample_indices = list()
            pid_list = np.random.choice(self.dataset.label_set, 
                self.batch_size[0], replace=False)
            for pid in pid_list:
                _index = self.dataset.index_dict[pid]
                if len(_index) >= self.batch_size[1]:
                    _index = np.random.choice(_index, self.batch_size[1], replace=False).tolist()
                else:
                    _index = np.random.choice(_index, self.batch_size[1], replace=True).tolist()             
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size

class DistributedTripletSampler(tordata.sampler.Sampler):
    def __init__(self, dataset, batch_size, world_size=None, rank=None, random_seed=2019):
        np.random.seed(random_seed)
        random.seed(random_seed)
        print("random_seed={} for DistributedTripletSampler".format(random_seed))
        
        if world_size is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            world_size = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.batch_size = batch_size
        self.world_size = world_size
        self.rank = rank
        self.random_seed = 0
        
        self.batch_ids_per_world = int(math.ceil(self.batch_size[0] * 1.0 / self.world_size))
        assert(self.batch_size[0] % self.world_size == 0)
        self.total_batch_per_world = int(math.ceil(len(self.dataset.label_set) * 1.0 / self.batch_size[0]))

    def __iter__(self):
        while (True):
            g = torch.Generator()
            g.manual_seed(self.random_seed)
            pid_index_all_world = torch.randperm(len(self.dataset.label_set), generator=g).tolist()
            pid_index_cur_world = pid_index_all_world[self.rank:len(self.dataset.label_set):self.world_size]
            # if self.rank == 0:
            #     print("random_seed={}".format(self.random_seed))
            #     print("pid_index_all_world={}, pid_index_cur_world={}".format(pid_index_all_world, pid_index_cur_world))
            #     print("batch_ids_per_world={}, total_batch_per_world={}".format(self.batch_ids_per_world, self.total_batch_per_world))
            
            sample_indices = list()
            # pid_index_cur_batch = random.sample(pid_index_cur_world, self.batch_ids_per_world)
            pid_index_cur_batch = np.random.choice(pid_index_cur_world, self.batch_ids_per_world, replace=False)
            for pid_index in pid_index_cur_batch:
                _index = self.dataset.index_dict[self.dataset.label_set[pid_index]]
                # _index = random.choices(_index, k=self.batch_size[1])
                if len(_index) >= self.batch_size[1]:
                    _index = np.random.choice(_index, self.batch_size[1], replace=False).tolist()
                else:
                    _index = np.random.choice(_index, self.batch_size[1], replace=True).tolist() 
                sample_indices += _index
            yield sample_indices

    def __len__(self):
        return self.dataset.data_size

    def set_random_seed(self, seed):
        self.random_seed = seed