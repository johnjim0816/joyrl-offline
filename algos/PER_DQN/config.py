#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-02-21 20:32:11
LastEditor: JiangJi
LastEditTime: 2023-05-18 23:14:20
Discription: 
'''
from config.config import DefaultConfig

class AlgoConfig(DefaultConfig):
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.95 # epsilon start value
        self.epsilon_end = 0.01 # epsilon end value
        self.epsilon_decay = 500 # epsilon decay rate
        self.gamma = 0.95 # discount factor
        self.lr = 0.0001 # learning rate
        self.buffer_type = 'PER_QUE' # replay buffer type
        self.buffer_size = 100000 # size of replay buffer
        self.per_alpha = 0.6 # alpha for prioritized replay buffer
        self.per_beta = 0.4 # beta for prioritized replay buffer
        self.per_beta_annealing = 0.001 # beta increment for prioritized replay buffer
        self.per_epsilon = 0.01 # epsilon for prioritized replay buffer
        self.batch_size = 64 # batch size
        self.target_update = 4 # target network update frequency
        # value network layers config
        self.value_layers = [
            {'layer_type': 'Linear', 'layer_size': [64], 'activation': 'ReLU'},
            {'layer_type': 'Linear', 'layer_size': [64], 'activation': 'ReLU'},
        ]
