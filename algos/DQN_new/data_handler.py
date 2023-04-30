#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-23 00:55:32
LastEditor: JiangJi
LastEditTime: 2023-04-29 15:23:10
Discription: 
'''
import ray
import torch
import numpy as np
from algos.base.buffers import BufferCreator


class DataHandler:
    def __init__(self, cfg):
        self.cfg = cfg
        self.buffer = BufferCreator(cfg)()
    def add_transition(self, transition):
        exp = self.create_exp(transition)
        self.buffer.push(exp)
    def sample_training_data(self):
        exps = self.buffer.sample(self.cfg.batch_size)
        if exps is not None:
            return self.handle_exps_before_update(exps)
        else:
            return None
    def create_exp(self,transtion):
        state, action, reward, next_state, terminated, info = transtion
        algo_name = self.cfg.algo_name
        exp_mod = __import__(f"algos.{algo_name}.exp", fromlist=['Exp'])
        exp = exp_mod.Exp(state = state, action = action, reward = reward, next_state = next_state, done = terminated, info = info)
        return [exp]
    def handle_exps_before_update(self, exps):
        # 将采样的数据转换为 tensor
        states = np.array([exp.state for exp in exps])
        actions = np.array([exp.action for exp in exps])
        rewards = np.array([exp.reward for exp in exps])
        next_states = np.array([exp.next_state for exp in exps])
        dones = np.array([exp.done for exp in exps])
        data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones}
        return data
