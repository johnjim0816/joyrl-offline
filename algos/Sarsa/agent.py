#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-03-10 23:35:32
LastEditor: JiangJi
LastEditTime: 2023-03-11 00:13:55
Discription: 
'''
from collections import defaultdict
import numpy as np
import math
import torch

class Agent(object):
    def __init__(self,cfg):
        self.n_actions = cfg.n_actions 
        self.lr = cfg.lr 
        self.gamma = cfg.gamma    
        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table  = defaultdict(lambda: np.zeros(self.n_actions)) # Q table
        self.sample_count = 0  
    def sample_action(self, state):
        ''' another way to represent e-greedy policy
        '''
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) # The probability to select a random action, is is log decayed
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)]) # choose action corresponding to the maximum q value
        else:
            action = np.random.choice(self.n_actions) # choose action randomly
        return action
    def predict_action(self,state):
        ''' predict action while testing 
        '''
        action = np.argmax(self.Q_table[str(state)])
        return action
    def update(self, state, action, reward, next_state, next_action, done):
        Q_predict = self.Q_table[str(state)][action]
        if done:
            Q_target = reward  # terminal state
        else:
            Q_target = reward + self.gamma * self.Q_table[str(next_state)][next_action] # the only difference from Q learning
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict) 
    def save_model(self,fpath):
        import dill
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(
            obj=self.Q_table,
            f=f"{fpath}/checkpoint.pkl",
            pickle_module=dill
        )
    def load_model(self, fpath):
        import dill
        self.Q_table=torch.load(f=f"{fpath}/checkpoint.pkl",pickle_module=dill)