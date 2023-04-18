#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 22:30:46
LastEditor: JiangJi
LastEditTime: 2023-04-18 13:12:20
Discription: 
'''
import torch.nn as nn
from algos.base.layers import create_layer

class ValueNetwork(nn.Module):
    def __init__(self, cfg) -> None:
        super(ValueNetwork, self).__init__()
        self.layers_cfg = cfg.value_layers # load layers config
        self.layers = nn.ModuleList()
        for layer_cfg in self.layers_cfg:
            layer_type, layer_dim, act_name = layer_cfg['layer_type'].lower(), layer_cfg['layer_dim'], layer_cfg['activation'].lower()
            in_dim, out_dim = layer_dim
            if in_dim == 'n_states':
                in_dim = cfg.n_states
            if out_dim == 'n_actions':
                out_dim = cfg.n_actions
            self.layers.append(create_layer(layer_type,in_dim, out_dim,act_name))
        # self.layers = self.layers.to(cfg.device)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
if __name__ == "__main__":
    # 调试用：export PYTHONPATH=./:$PYTHONPATH
    import torch
    from config.config import MergedConfig
    cfg = MergedConfig()
    cfg.n_states = [None,4]
    cfg.n_actions = 2
    cfg.value_layers = [
        {'layer_type': 'Linear', 'layer_dim': ['n_states', 64], 'activation': 'ReLU'},
        {'layer_type': 'Linear', 'layer_dim': [64, 64], 'activation': 'ReLU'},
        {'layer_type': 'Linear', 'layer_dim': [64, 1], 'activation': 'None'}
    ]
    value_net = ValueNetwork(cfg)
    print(value_net)
    x = torch.randn(1,4)
    print(value_net(x))