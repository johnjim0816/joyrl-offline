#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 22:30:46
LastEditor: JiangJi
LastEditTime: 2023-04-19 01:54:18
Discription: 
'''
import sys, os
sys.path.append(os.getcwd())
import torch.nn as nn
from algos.base.layers import create_layer

class ValueNetwork(nn.Module):
    def __init__(self, cfg, input_size, action_dim) -> None:
        super(ValueNetwork, self).__init__()
        self.layers_cfg = cfg.value_layers # load layers config
        self.layers = nn.ModuleList()
        output_size = input_size
        for layer_cfg in self.layers_cfg:
            layer_type, layer_dim, act_name = layer_cfg['layer_type'].lower(), layer_cfg['layer_dim'], layer_cfg['activation'].lower()
            layer, layer_out_dim = create_layer(layer_type, output_size, layer_dim, act_name)
            output_size = [None, layer_out_dim]
            self.layers.append(layer)  
        action_layer, action_out_dim = create_layer('linear', output_size, [action_dim], 'none')
        self.layers.append(action_layer)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
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
    value_net = ValueNetwork(cfg, state_dim)
    print(value_net)
    x = torch.randn(1,4)
    print(value_net(x))