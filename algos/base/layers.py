#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 22:30:15
LastEditor: JiangJi
LastEditTime: 2023-04-18 13:15:57
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
    in_dim = in_dim[-1]
    return nn.Sequential(nn.Linear(in_dim,out_dim),activation_dics[act_name]())
def dense_layer(in_dim,out_dim,act_name='relu'):
    """ 生成一个全连接层
        layer_dim: 全连接层的输入输出维度
        activation: 激活函数
    """
    
def conv2d_layer(in_channel, out_channel, kernel_size, stride, padding, act_name='relu'):
    """ 生成一个卷积层
        layer_dim: 卷积层的输入输出维度
        activation: 激活函数
    """
    padding = 'same' if stride == 1 else 'valid'
    return nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding),activation_dics[act_name]() )
def create_layer(layer_type, in_dim,out_dim, act_name='relu'):
    """ 生成一个层
        layer_type: 层的类型
        layer_dim: 层的输入输出维度
        activation: 激活函数
    """
    if layer_type == "linear":
        return linear_layer(in_dim,out_dim,act_name)
    elif layer_type == "conv2d":
        return conv2d_layer(in_dim,out_dim,act_name)
    else:
        raise NotImplementedError