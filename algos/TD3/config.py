#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-02-20 17:22:28
LastEditor: JiangJi
LastEditTime: 2023-04-16 21:25:10
Discription: 
'''
class AlgoConfig:
    def __init__(self) -> None:
        self.action_type = 'dpg' # action type, dpg: deterministic policy gradient
        self.buffer_type = 'REPLAY_QUE' # replay buffer type
        self.explore_steps = 100  # exploration steps before training
        self.policy_freq = 2  # policy update frequency
        self.actor_lr = 1e-4 # actor learning rate 3e-4
        self.critic_lr = 1e-3 # critic learning rate
        self.gamma = 0.99 # discount factor
        self.tau = 0.005 # target smoothing coefficient
        self.policy_noise = 0.2 # noise added to target policy during critic update
        self.expl_noise = 0.1 # std of Gaussian exploration noise
        self.noise_clip = 0.5 # range to clip target policy noise
        self.batch_size = 100 # batch size for both actor and critic
        self.buffer_size = 1000000 # replay buffer size

        self.global_best_reward = -1800
        self.actor_layers = [
            {'layer_type': 'Linear', 'layer_size': [200], 'activation': 'ReLU'},
            {'layer_type': 'Linear', 'layer_size': [200], 'activation': 'ReLU'},
        ]
        self.critic_layers = [
            {'layer_type': 'Linear', 'layer_size': [200], 'activation': 'ReLU'},
            {'layer_type': 'Linear', 'layer_size': [200], 'activation': 'ReLU'},
        ]