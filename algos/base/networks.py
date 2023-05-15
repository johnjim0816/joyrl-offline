#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 22:30:46
LastEditor: JiangJi
LastEditTime: 2023-04-26 00:01:36
Discription: 
'''
import sys, os
sys.path.append(os.getcwd())
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
    
if __name__ == "__main__":
    # 调试用：export PYTHONPATH=./:$PYTHONPATH
    import torch
    from config.config import MergedConfig
    cfg = MergedConfig()
    state_dim = [None]
    cfg.n_actions = 2
    cfg.value_layers = [
        {'layer_type': 'embed', 'n_embeddings': 10, 'embedding_dim': 32, 'activation': 'none'},
        {'layer_type': 'Linear', 'layer_dim': [64], 'activation': 'ReLU'},
        {'layer_type': 'Linear', 'layer_dim': [64], 'activation': 'ReLU'},
    ]
    value_net = ValueNetwork(cfg, state_dim, cfg.n_actions)
    print(value_net)
    x = torch.tensor([36])
    print(x.shape)
    print(value_net(x))