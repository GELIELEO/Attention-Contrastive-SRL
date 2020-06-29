import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
import scipy.signal
import numpy as np


class RolloutBuffer:
    """
    A buffer for storing observation(augmented) using random policy.
    """

    def __init__(self, obs_dim, size, num_envs, device=torch.device('cpu')):
        self.obs_dim = obs_dim
        self.obs_buf_1 = torch.zeros((size, *obs_dim), dtype=torch.float32).to(device)
        self.obs_buf_2 = torch.zeros((size, *obs_dim), dtype=torch.float32).to(device)
        # self.act_buf = torch.zeros(size, dtype=torch.int8).to(device)
        # self.rst_buf = torch.zeros(size, dtype=torch.float32).to(device) # 0 for no collision, 1 for collision

        # to control the indexing
        self.ptr = torch.zeros(num_envs,dtype=torch.int).to(device)
        self.path_start_idx = torch.zeros(num_envs,dtype=torch.int).to(device)

        # constants
        self.max_size, self.block_size = size, size//num_envs

        # device
        self.device = device

    def share_memory(self):
        self.obs_buf_1.share_memory_()
        self.obs_buf_2.share_memory_()
        # self.act_buf.share_memory_()
        # self.rst_buf.share_memory_()
        self.ptr.share_memory_()
        

    def store(self, envid, obs_1, obs_2):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr[envid].item()  < self.block_size  # buffer has to have room so you can store
        ptr = self.ptr[envid].item()+ envid * self.block_size
        self.obs_buf_1[ptr].copy_(obs_1)
        self.obs_buf_2[ptr].copy_(obs_2)
        # self.act_buf[ptr].copy_(act)
        # self.rst_buf[ptr].copy_(result)
        
        self.ptr[envid] += 1

    def finish_path(self, envid):
        self.path_start_idx[envid] = self.ptr[envid]

    
    def batch_generator(self, batch_size, num_steps=1):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer. Also, resets some pointers in the buffer.
        """
        if self.ptr.sum().item() != 0:
            assert self.ptr.sum().item() == self.max_size, f'expected size:{self.max_size}, actual:{self.ptr.sum().item()}' 
            self.ptr.copy_(torch.zeros_like(self.ptr))
            self.path_start_idx.copy_(torch.zeros_like(self.path_start_idx))
        num_blocks = self.max_size//num_steps
        indice = torch.arange(self.max_size).view(-1,num_steps)
        batch_sampler = BatchSampler( SubsetRandomSampler(range(num_blocks)), batch_size//num_steps, drop_last=False)
        for block in batch_sampler:
            idx = indice[block].view(-1)
            yield [
                self.obs_buf_1[idx], self.obs_buf_2[idx]
            ]
