#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-23 00:54:59
LastEditor: JiangJi
LastEditTime: 2023-04-26 23:21:42
Discription: 
'''
import random
import math
import ray
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from algos.base.agents import BaseAgent
from algos.base.buffers import BufferCreator
from algos.base.agents import BaseAgent
from algos.base.networks import ValueNetwork

@ray.remote
class Agent(BaseAgent):
    def __init__(self,cfg) -> None:
        super(Agent, self).__init__(cfg)
        self.cfg = cfg
        self.obs_space = cfg.obs_space
        self.action_space = cfg.action_space
        self.device = torch.device(cfg.device) 
        self.gamma = cfg.gamma  
        # e-greedy 策略相关参数
        self.sample_count = 0  # 采样动作计数
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update
        self.memory = BufferCreator(cfg)()
        self.update_step = 0
        self.create_graph()
    def create_graph(self):
        self.state_size = [None, self.obs_space.shape[0]]
        action_dim = self.action_space.n
        self.policy_net = ValueNetwork(self.cfg, self.state_size, action_dim).to(self.device)
        self.target_net = ValueNetwork(self.cfg, self.state_size, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # or use this to copy parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.lr) 
       
    def create_optm(self):
        self.optm = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        
    def sample_action(self, state):
        ''' 采样动作
        Args:
            state (array): 状态
        Returns:
            action (int): 动作
        '''
        self.sample_count += 1
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            action = self.predict_action(state)
        else:
            action = self.action_space.sample()
        return action
    
    def predict_action(self,state):
        ''' 预测动作
        Args:
            state (array): 状态
        Returns:
            action (int): 动作
        '''
        with torch.no_grad():
            state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        return action
    def update(self, **kwargs):
        # 从 replay buffer 中采样
        states = kwargs.get('states')
        actions = kwargs.get('actions')
        next_states = kwargs.get('next_states')
        rewards = kwargs.get('rewards')
        dones = kwargs.get('dones')
        # 计算当前状态的 Q 值
        q_values = self.policy_net(states).gather(1, actions)
        # 计算下一个状态的最大 Q 值
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(dim=1)
        # 计算目标 Q 值
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        # 计算损失
        self.loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        self.loss.backward()
        # clip 防止梯度爆炸
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        if self.sample_count % self.target_update == 0: # 每 C 步更新一次 target_net
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.update_step += 1
        self.update_summary()
 