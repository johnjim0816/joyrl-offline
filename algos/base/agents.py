#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-17 22:40:10
LastEditor: JiangJi
LastEditTime: 2023-05-07 22:44:18
Discription: 
'''
import torch
import torch.nn as nn
class BasePolicy(nn.Module):
    def __init__(self,cfg) -> None:
        super().__init__()
        self.cfg = cfg
    def get_params(self):
        named_params_dict = dict(self.named_parameters())
        return named_params_dict
    def load_params(self, params_dict):
        self.load_state_dict(params_dict)
    def get_action(self, state):
        raise NotImplementedError
    def predict_action(self, state):
        raise NotImplementedError
    def create_summary(self):
        raise NotImplementedError
    def update_summary(self):
        ''' 更新 tensorboard 数据
        '''
        raise NotImplementedError
    def update(self):
        raise NotImplementedError
    def save(self):
        pass
    def load(self):
        pass