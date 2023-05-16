#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-16 16:12:07
LastEditor: JiangJi
LastEditTime: 2023-05-16 20:07:55
Discription: 
'''
import torch.nn as nn

class BaseActionLayer(nn.Module):
    def __init__(self, layer_type, layer_dim, activation):
        super(BaseActionLayer, self).__init__()
        pass

class DiscreteActionLayer(BaseActionLayer):
    def __init__(self, input_size, action_space, id = 0):
        super(DiscreteActionLayer, self).__init__()
        self.id = id
        self.action_space = action_space
        self.output_size = action_space.n
        self.softmax = nn.Softmax(dim=1)
        
class ContinousActionLayer(BaseActionLayer):
    def __init__(self, input_size, action_space, id = 0):
        super(ContinousActionLayer, self).__init__()
        self.id = id
        self.action_space = action_space
        self.output_size = action_space.shape[0]
        self.tanh = nn.Tanh()
        