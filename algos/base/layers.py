#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 22:30:15
LastEditor: JiangJi
LastEditTime: 2023-04-24 15:12:17
Discription: 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
activation_dics = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'softmax': nn.Softmax,'none': nn.Identity}  

class LayerConfig:
    ''' 层的配置类
    '''
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

def get_out_size_with_batch(layers,input_size,dtype=torch.float):
    """ 获取层的输出维度
        layer: 层
        in_dim: 层的输入维度
    """
    with torch.no_grad():
        x = torch.randn(10,*input_size[-1],dtype=dtype)
        out = layers(x)
    return [None,list(out.size())[1:] ]        
def embedding_layer(input_size, layer_cfg: LayerConfig):
    n_embeddings = layer_cfg.n_embeddings
    embedding_dim = layer_cfg.embedding_dim
    class EmbeddingLayer(nn.Module):
        def __init__(self, n_embeddings, embedding_dim):
            super(EmbeddingLayer, self).__init__()
            self.layer = nn.Embedding(num_embeddings=n_embeddings[0], embedding_dim=embedding_dim[0])

        def forward(self, x: torch.Tensor):
            # if x.dtype != torch.int:
            #     x = x.int()
            return self.layer(x)
    layer = EmbeddingLayer(n_embeddings, embedding_dim)
    output_size = get_out_size_with_batch(layer, input_size=input_size, dtype=torch.long)
    return layer, output_size 
def linear_layer(input_size,layer_cfg: LayerConfig):
    """ 生成一个线性层
        layer_size: 线性层的输入输出维度
        activation: 激活函数
    """
    layer_size = layer_cfg.layer_dim
    act_name = layer_cfg.activation.lower()
    in_dim = input_size[-1]
    out_dim = layer_size[0]
    layer = nn.Sequential(nn.Linear(in_dim,out_dim),activation_dics[act_name]())
    return layer, [None, out_dim]
def dense_layer(in_dim,out_dim,act_name='relu'):
    """ 生成一个全连接层
        layer_size: 全连接层的输入输出维度
        activation: 激活函数
    """
    
def conv2d_layer(in_channel, out_channel, kernel_size, stride, padding, act_name='relu'):
    """ 生成一个卷积层
        layer_size: 卷积层的输入输出维度
        activation: 激活函数
    """
    padding = 'same' if stride == 1 else 'valid'
    return nn.Sequential(nn.Conv2d(in_channel,out_channel,kernel_size,stride,padding),activation_dics[act_name]() )
def create_layer(input_size: list, layer_cfg: LayerConfig):
    """ 生成一个层
        layer_type: 层的类型
        layer_size: 层的输入输出维度
        activation: 激活函数
    """
    layer_type = layer_cfg.layer_type.lower()
    if layer_type == "linear":
        return linear_layer(input_size, layer_cfg)
    elif layer_type == "conv2d":
        return conv2d_layer(input_size, layer_cfg)
    elif layer_type == "embed":
        return embedding_layer(input_size, layer_cfg)
    else:
        raise NotImplementedError