#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 22:30:46
LastEditor: JiangJi
LastEditTime: 2023-05-17 00:43:09
Discription: 
'''
import torch.nn as nn
from algos.base.base_layers import create_layer, LayerConfig
from algos.base.action_layers import DiscreteActionLayer, ContinousActionLayer
class BaseNework(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class QNetwork(BaseNework):
    ''' Q network, for value-based methods like DQN
    '''
    def __init__(self, cfg, state_size, action_size):
        super(QNetwork, self).__init__()
        self.cfg = cfg
        self.dueling = hasattr(cfg, 'dueling') and cfg.dueling
        self.layers_cfg_dic = cfg.value_layers # load layers config
        self.layers = nn.ModuleList()
        output_size = state_size
        for layer_cfg_dic in self.layers_cfg_dic:
            if "layer_type" not in layer_cfg_dic:
                raise ValueError("layer_type must be specified in layer_cfg")
            layer_cfg = LayerConfig(**layer_cfg_dic)
            layer, layer_out_size = create_layer(output_size, layer_cfg)
            output_size = layer_out_size
            self.layers.append(layer) 
        action_dim = action_size[0]
        if self.dueling:
            # state value
            state_value_layer_cfg = LayerConfig(layer_type='linear', layer_dim=[1], activation='none')
            self.state_value_layer, layer_out_size = create_layer(output_size, state_value_layer_cfg)
            # action value
            action_value_layer_cfg = LayerConfig(layer_type='linear', layer_dim=[action_dim], activation='none')
            self.action_value_layer, layer_out_size = create_layer(output_size, action_value_layer_cfg)
        else:
            action_layer_cfg = LayerConfig(layer_type='linear', layer_dim=[action_dim], activation='none')
            self.action_value_layer, layer_out_size = create_layer(output_size, action_layer_cfg)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        if self.dueling:
            state_value = self.state_value_layer(x)
            action_value = self.action_value_layer(x)
            q_value = state_value + action_value - action_value.mean(dim=1, keepdim=True)
        else:
            q_value = self.action_value_layer(x)
        return q_value
        
class ValueNetwork(BaseNework):
    ''' Value network, for policy-based methods like DDPG, in which the actor and critic share the same network
    '''
    def __init__(self, cfg, state_size):
        super(ValueNetwork, self).__init__()
        pass

class BaseActorNetwork(nn.Module):
    def __init__(self) -> None:
        pass
class BaseCriticNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()

class ActorNetwork(BaseActorNetwork):
    def __init__(self, cfg, state_size, action_space) -> None:
        super().__init__()
        self.cfg = cfg
        self.continous = action_space.continous
        self.layers_cfg_dic = cfg.actor_layers # load layers config
        self.layers = nn.ModuleList()
        output_size = state_size
        for layer_cfg_dic in self.layers_cfg_dic:
            if "layer_type" not in layer_cfg_dic:
                raise ValueError("layer_type must be specified in layer_cfg")
            layer_cfg = LayerConfig(**layer_cfg_dic)
            layer, layer_out_size = create_layer(output_size, layer_cfg)
            output_size = layer_out_size
            self.layers.append(layer) 
        if self.continous:
            self.action_layer = ContinousActionLayer(cfg, output_size, action_space)
        else:
            self.action_layer = DiscreteActionLayer(cfg, output_size, action_space)
    def forward(self, x, legal_actions=None):
        for layer in self.layers:
            x = layer(x)
        if self.continous:
            mu, sigma = self.action_layer(x)
            return mu, sigma
        else:
            probs = self.action_layer(x, legal_actions)
            return probs

class CriticNetwork(BaseCriticNetwork):
    def __init__(self, cfg, state_size):
        super(CriticNetwork, self).__init__()
        self.cfg = cfg
        self.layers_cfg_dic = cfg.critic_layers # load layers config
        self.layers = nn.ModuleList()
        output_size = state_size
        for layer_cfg_dic in self.layers_cfg_dic:
            if "layer_type" not in layer_cfg_dic:
                raise ValueError("layer_type must be specified in layer_cfg")
            layer_cfg = LayerConfig(**layer_cfg_dic)
            layer, layer_out_size = create_layer(output_size, layer_cfg)
            output_size = layer_out_size
            self.layers.append(layer) 
        head_layer_cfg = LayerConfig(layer_type='linear', layer_dim=[1], activation='none')
        self.head_layer, layer_out_size = create_layer(output_size, head_layer_cfg)
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        value = self.head_layer(x)
        return value
        

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
    value_net = QNetwork(cfg, state_dim, cfg.n_actions)
    print(value_net)
    x = torch.tensor([36])
    print(x.shape)
    print(value_net(x))