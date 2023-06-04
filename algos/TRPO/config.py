#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-21 21:21:20
LastEditor: JiangJi
LastEditTime: 2023-06-04 16:57:47
Discription: 
'''
class AlgoConfig:
    def __init__(self):
        self.gamma = 0.98 # discount factor
        self.lmbda = 0.95 # GAE参数
        self.alpha = 0.5
        self.kl_constraint = 0.0005
        
        self.critic_lr = 0.02 # learning rate for critic
        self.eps_clip = 0.2 # clip parameter for PPO
        self.search_steps = 15 # steps for the line search
        self.grad_steps = 10 # steps to calculate the conjugate gradients
        self.train_batch_size = 64 # ppo train batch size
        self.actor_hidden_dim = 128 # hidden dimension for actor
        self.critic_hidden_dim = 128 # hidden dimension for critic
        self.search_steps = 15
        self.grad_steps = 10        
        self.min_policy = 0 # min value for policy (for discrete action space)
