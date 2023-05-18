#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-19 00:16:03
LastEditor: JiangJi
LastEditTime: 2023-05-19 00:17:13
Discription: 
'''
import torch
import torch.nn as nn
import math,random
import numpy as np
from algos.base.policies import ToyPolicy

class Policy(ToyPolicy):
    def __init__(self,cfg) -> None:
        super(Policy, self).__init__(cfg)
        self.Q_table = None
    def get_action(self, state, mode='sample', **kwargs):
        return super().get_action(state, mode, **kwargs)
    def sample_action(self, state, **kwargs):
        raise NotImplementedError
    def predict_action(self, state, **kwargs):
        raise NotImplementedError
    def train(self, **kwargs):
        raise NotImplementedError
