#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-12-24 20:41:56
LastEditor: JiangJi
LastEditTime: 2023-03-12 18:32:17
Discription: 
'''
from config.general_config import DefaultConfig

import torch
class AlgoConfig(DefaultConfig):
    def __init__(self):
        self.gamma = 0.99 # discount factor
        self.tau = 1.0 # 1.0 means hard update
        self.hidden_dim = 256 # hidden_dim for MLP
        self.Vmin = 0. # support of C51  
        self.Vmax = 200. # support of C51 
        self.n_atoms = 51 # support of C51
        self.support = torch.linspace(self.Vmin, self.Vmax, self.n_atoms) # support of C51
        self.delta_z = (self.Vmax - self.Vmin) / (self.n_atoms - 1) # support of C51

        self.n_step = 1 #the n_step for N-step DQN
        self.batch_size = 32 # batch size
        self.lr = 0.0001 # learning rate
        self.target_update = 200 # target network update frequency
        self.memory_capacity = 10000 # size of replay buffer
        self.epsilon_start = 0.95  # epsilon start value
        self.epsilon_end = 0.01  # epsilon end value
        self.epsilon_decay = 500  # epsilon decay rate