#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-17 01:08:36
LastEditor: JiangJi
LastEditTime: 2023-05-17 13:34:56
Discription: 
'''
import numpy as np
from algos.base.data_handlers import BaseDataHandler
class DataHandler(BaseDataHandler):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.train_batch_size = cfg.train_batch_size
    def sample_training_data(self):
        ''' sample training data from buffer
        '''
        if len(self.buffer) < self.train_batch_size:
            return None
        exps = self.buffer.sample()
        if exps is not None:
            return self.handle_exps_before_update(exps)
        else:
            return None
    def handle_exps_before_update(self, exps):
        ''' convert exps to training data
        '''
        states = np.array([exp.state for exp in exps])
        actions = np.array([exp.action for exp in exps])
        rewards = np.array([exp.reward for exp in exps])
        next_states = np.array([exp.next_state for exp in exps])
        dones = np.array([exp.done for exp in exps])
        probs = [exp.probs for exp in exps]
        log_probs = np.array([exp.log_probs for exp in exps])
        data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones, 'probs': probs, 'log_probs': log_probs}
        return data