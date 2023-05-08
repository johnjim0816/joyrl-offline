#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-17 22:40:10
LastEditor: JiangJi
LastEditTime: 2023-05-09 13:01:57
Discription: 
'''
import torch
import torch.nn as nn
class BasePolicy(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.optimizer = None
    def get_policy_params(self):
        named_params_dict = dict(self.named_parameters())
        return named_params_dict
    def load_policy_params(self, params_dict):
        self.load_state_dict(params_dict)
    def get_optimizer_params(self):
        return self.optimizer.state_dict()
    def load_optimizer_params(self, optim_params_dict):
        self.optimizer.load_state_dict(optim_params_dict)
    def get_action(self, state):
        raise NotImplementedError
    def predict_action(self, state):
        raise NotImplementedError
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
    def save(self):
        pass
    def load(self):
        pass