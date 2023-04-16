#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 22:30:15
LastEditor: JiangJi
LastEditTime: 2023-04-16 22:30:41
Discription: 
'''
import torch.nn as nn
import torch.nn.functional as F
activation_dics = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,'none': nn.Identity}             
def linear_layer(in_dim,out_dim,act_name='relu'):
    """ 生成一个线性层
        layer_dim: 线性层的输入输出维度
        activation: 激活函数
    """
    return nn.Sequential(nn.Linear(in_dim,out_dim),activation_dics[act_name]())

def create_layer(layer_type, in_dim,out_dim, act_name='relu'):
    """ 生成一个层
        layer_type: 层的类型
        layer_dim: 层的输入输出维度
        activation: 激活函数
    """
    if layer_type == "linear":
        return linear_layer(in_dim,out_dim, act_name)
    else:
        raise ValueError("layer_type must be linear")