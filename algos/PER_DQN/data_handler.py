#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-18 22:41:08
LastEditor: JiangJi
LastEditTime: 2023-05-18 23:17:19
Discription: 
'''
import numpy as np
from algos.base.data_handler import BaseDataHandler
class DataHandler(BaseDataHandler):
    def __init__(self, cfg):
        super().__init__(cfg)
    def sample_training_data(self):
        ''' sample training data from buffer
        '''
        exps, idxs, weights = self.buffer.sample()
        if exps is not None:
            return self.handle_exps_before_train(exps, idxs = idxs, weights = weights)
        else:
            return None
    def handle_exps_before_train(self, exps, **kwargs):
        ''' convert exps to training data
        '''
        states = np.array([exp.state for exp in exps])
        actions = np.array([exp.action for exp in exps])
        rewards = np.array([exp.reward for exp in exps])
        next_states = np.array([exp.next_state for exp in exps])
        dones = np.array([exp.done for exp in exps])
        idxs = kwargs.get('idxs')
        weights = kwargs.get('weights')
        data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones, 'idxs': idxs, 'weights': weights}
        return data
    def handle_exps_after_update(self):
        ''' handle exps after update
        '''
        idxs, td_errors = self.data_after_train['idxs'], self.data_after_train['td_errors']
        self.buffer.update_priorities(idxs, td_errors)