#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 22:30:46
LastEditor: JiangJi
LastEditTime: 2023-04-24 15:12:45
Discription: 
'''
import sys, os
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from algos.base.layers import create_layer, LayerConfig

class ValueNetwork(nn.Module):
    def __init__(self, cfg, input_size, action_dim) -> None:
        super(ValueNetwork, self).__init__()
        self.layers_cfg_dic = cfg.value_layers # load layers config
        self.layers = nn.ModuleList()
        output_size = input_size
        for layer_cfg_dic in self.layers_cfg_dic:
            if "layer_type" not in layer_cfg_dic:
                raise ValueError("layer_type must be specified in layer_cfg")
            layer_cfg = LayerConfig(**layer_cfg_dic)
            layer, layer_out_size = create_layer(output_size, layer_cfg)
            output_size = layer_out_size
            self.layers.append(layer)  
        action_layer_cfg = LayerConfig(layer_type='linear', layer_dim=[action_dim], activation='none')
        action_layer, layer_out_size = create_layer(output_size, action_layer_cfg)
        self.layers.append(action_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def create_critic(input_size,output_size,layers_dict):
    layers = nn.ModuleList()
    for layer_cfg_dic in layers_dict:
        if "layer_type" not in layer_cfg_dic:
            raise ValueError("layer_type must be specified in layer_cfg")
        layer_cfg = LayerConfig(**layer_cfg_dic)
        layer, input_size = create_layer(input_size, layer_cfg)
        layers.append(layer)
    action_layer_cfg = LayerConfig(layer_type='linear', layer_dim=[output_size], activation='none')
    action_layer, _ = create_layer(input_size, action_layer_cfg)
    layers.append(action_layer)
    return layers

class CriticNetwork(nn.Module):
    def __init__(self, cfg, input_size, action_dim) -> None:
        super(CriticNetwork, self).__init__()
        self.layers_cfg_dic = cfg.value_layers  # load layers config
        self.q_layers = create_critic([None,input_size+action_dim], 1, self.layers_cfg_dic)
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        for layer in self.q_layers:
            x = layer(x)
        return x

class DoubleCriticNetwork(nn.Module):
    def __init__(self, cfg, input_size, action_dim) -> None:

        super(DoubleCriticNetwork, self).__init__()
        self.layers_cfg_dic = cfg.value_layers  # load layers config
        self.q1_layers = create_critic([None,input_size+action_dim], 1,self.layers_cfg_dic)
        self.q2_layers = create_critic([None,input_size+action_dim], 1, self.layers_cfg_dic)

    def forward(self, state, action):
        x_orig = torch.cat([state,action],dim=-1)
        x1 = x_orig[:]
        x2 = x_orig[:]

        for layer in self.q1_layers:
            x1 = layer(x1)
        for layer in self.q2_layers:
            x2 = layer(x2)
        return x1,x2

    def q1(self,state,action):
        x1 = torch.cat([state, action], dim=-1)
        for layer in self.q1_layers:
            x1 = layer(x1)
        return x1

    def q_all(self,state,action,with_var=False):
        q1,q2 = self.forward(state,action)
        q_all = torch.cat( [q1.unsqueeze(0),q2.unsqueeze(0)], dim=0 )
        if with_var:
            std_q = torch.std(q_all,dim=0,keepdim=False,unbiased=False)
            return q_all,std_q
        return q_all


if __name__ == "__main__":
    # 调试用：export PYTHONPATH=./:$PYTHONPATH
    import torch
    from config.config import MergedConfig
    cfg = MergedConfig()
    state_dim = [None,4]
    cfg.n_actions = 2
    cfg.value_layers = [
        {'layer_type': 'Linear', 'layer_dim': [64], 'activation': 'ReLU'},
        {'layer_type': 'Linear', 'layer_dim': [64], 'activation': 'ReLU'},
    ]
    value_net = ValueNetwork(cfg, state_dim, cfg.n_actions)
    print(value_net)
    x = torch.randn(1,4)
    print(value_net(x))

    print("------ check CriticNetwork & DoubleCriticNetwork -----------")
    critic_net = CriticNetwork(cfg, 4, cfg.n_actions)
    double_critic_net = DoubleCriticNetwork(cfg, 4, cfg.n_actions)
    print(double_critic_net)
    state = torch.randn(1,4)
    action = torch.randn(1,2)
    print(critic_net(state,action))
    print(double_critic_net.q_all(state,action,True))