#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-02-21 20:32:11
LastEditor: JiangJi
LastEditTime: 2023-05-16 00:09:55
Discription: 
'''
import numpy as np
from algos.base.buffers import BufferCreator
from algos.base.exps import Exp

class DataHandler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.buffer = BufferCreator(cfg)()

    def add_transition(self, transition):
        ''' add transition to buffer
        '''
        exp = self.create_exp(transition)
        self.buffer.push(exp)

    def sample_training_data(self):
        ''' sample training data from buffer
        '''
        exps = self.buffer.sample(self.cfg.batch_size)
        if exps is not None:
            return self.handle_exps_before_update(exps)
        else:
            return None
        
    def create_exp(self,transtion):
        ''' create experience
        '''
        state, action, reward, next_state, terminated, info = transtion
        exp = Exp(state = state, action = action, reward = reward, next_state = next_state, done = terminated, info = info)
        return [exp]
    
    def handle_exps_before_update(self, exps):
        ''' convert exps to training data
        '''
        states = np.array([exp.state for exp in exps])
        actions = np.array([exp.action for exp in exps])
        rewards = np.array([exp.reward for exp in exps])
        next_states = np.array([exp.next_state for exp in exps])
        dones = np.array([exp.done for exp in exps])
        data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones}
        return data
