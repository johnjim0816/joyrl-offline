#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 22:30:15
LastEditor: JiangJi
LastEditTime: 2023-04-19 02:07:32
Discription: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
activation_dics = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,'none': nn.Identity}  
def get_out_size_with_batch(layers,input_size,dtype=torch.float):
    """ 获取层的输出维度
        layer: 层
        in_dim: 层的输入维度
    """
    with torch.no_grad():
        x = torch.randn(10,*input_size[-1],dtype=dtype)
        out = layers(x)
    return [None,list(out.size())[1:]  ]         
def linear_layer(input_size,layer_dim,act_name='relu'):
    """ 生成一个线性层
        layer_dim: 线性层的输入输出维度
        activation: 激活函数
    """
    in_dim = input_size[-1]
    out_dim = layer_dim[0]
    layer = nn.Sequential(nn.Linear(in_dim,out_dim),activation_dics[act_name]())
    return layer,out_dim
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