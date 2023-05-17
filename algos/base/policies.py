#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-17 22:40:10
LastEditor: JiangJi
LastEditTime: 2023-05-17 12:57:40
Discription: 
'''
import torch
import torch.nn as nn
import torch.optim as optim
class BasePolicy(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.obs_space = cfg.obs_space
        self.action_space = cfg.action_space
        self.optimizer = None
        self.policy_transition = {}
        self.get_state_action_size()
    def get_state_action_size(self):
        self.state_size = [None, self.obs_space.shape[0]]
        self.action_size = [self.action_space.n]
        return self.state_size, self.action_size
    def create_optimizer(self):
        self.optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr) 
    def get_policy_params(self):
        named_params_dict = dict(self.named_parameters())
        return named_params_dict
    def load_policy_params(self, params_dict):
        self.load_state_dict(params_dict)
    def get_optimizer_params(self):
        return self.optimizer.state_dict()
    def load_optimizer_params(self, optim_params_dict):
        self.optimizer.load_state_dict(optim_params_dict)
    def get_action(self,state, sample_count = None, mode = 'sample'):
        ''' 
        获取动作
        '''
        if mode == 'sample':
            return self.sample_action(state, sample_count = sample_count)
        elif mode == 'predict':
            return self.predict_action(state)
        else:
            raise NotImplementedError
    def sample_action(self, state, sample_count = None):
        raise NotImplementedError
    def predict_action(self, state):
        raise NotImplementedError
    def update_policy_transition(self):
        self.policy_transition = {}
    def get_policy_transition(self):
        return self.policy_transition
    def create_summary(self):
        '''
        创建 tensorboard 数据
        '''
        self.summary = {
            'scalar': {
                'loss': 0.0,
            },
        }
    def update_summary(self):
        ''' 更新 tensorboard 数据
        '''
        self.summary['scalar']['loss'] = self.loss.item()
    def update(self):
        raise NotImplementedError
    def save_model(self, fpath):
        torch.save(self.state_dict(), fpath)
    def load_model(self, fpath):
        self.load_state_dict(torch.load(fpath))