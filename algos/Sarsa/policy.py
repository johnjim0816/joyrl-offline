#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-19 01:05:01
LastEditor: JiangJi
LastEditTime: 2023-05-19 01:06:23
Discription: 
'''

import torch
import torch.nn as nn
import math,random
import numpy as np
import pickle
from collections import defaultdict
from algos.base.policies import ToyPolicy

class Policy(ToyPolicy):
    def __init__(self,cfg) -> None:
        super(Policy, self).__init__(cfg)
        self.lr = cfg.lr 
        self.gamma = cfg.gamma 
        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table = defaultdict(lambda: np.zeros(self.n_actions))
        self.create_summary()
    def get_action(self, state, mode='sample', **kwargs):
        return super().get_action(state, mode, **kwargs)
    def sample_action(self, state, **kwargs):
        self.sample_count = kwargs.get('sample_count')
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)]) #  select the action with max Q value
        else:
            action = np.random.choice(self.n_actions) # random select an action
        return action
    def predict_action(self, state, **kwargs):
        action = np.argmax(self.Q_table[str(state)])
        return action
    def learn(self, **kwargs):
        state, action, reward, next_state, done = kwargs.get('state'), kwargs.get('action'), kwargs.get('reward'), kwargs.get('next_state'), kwargs.get('done')
        next_action = self.sample_action(next_state, sample_count=self.sample_count)
        Q_predict = self.Q_table[str(state)][action] 
        if done: 
            Q_target = reward 
        else:
            Q_target = reward + self.gamma * self.Q_table[str(next_state)][next_action] 
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)
        self.loss = (Q_target - Q_predict) ** 2
        self.update_summary() # update summary
    def save_model(self,fpath):
        ''' save model
        '''
        with open(f"{fpath}", 'wb') as f:
            pickle.dump(dict(self.Q_table), f)
    def load_model(self, fpath):
        ''' load model
        '''
        with open(fpath, 'rb') as f:
            self.Q_table = pickle.load(f)