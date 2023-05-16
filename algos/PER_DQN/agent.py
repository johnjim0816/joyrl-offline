#!/usr/bin/env python
# coding=utf-8
'''
Author: wangzhongren
Email: wangzhongren@sjtu.edu.cn
Date: 2022-11-20 17:20:00
LastEditor: wangzhongren
LastEditTime: 2023-03-31 23:21:52
Discription: 
'''

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import ray
import numpy as np
from common.layers import QNetwork
from common.memories import PrioritizedReplayBuffer,PrioritizedReplayBufferQue
from common.optms import SharedAdam

class Agent:
    def __init__(self, cfg, is_share_agent = False):

        self.n_actions = cfg.n_actions  
        self.device = torch.device(cfg.device) 
        self.gamma = cfg.gamma  
        ## e-greedy parameters
        self.sample_count = 0  # sample count for epsilon decay
        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update
        
        self.policy_net = QNetwork(cfg).to(self.device)
        self.target_net = QNetwork(cfg).to(self.device)
        ## copy parameters from policy net to target net
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): 
            target_param.data.copy_(param.data)
        # self.target_net.load_state_dict(self.policy_net.state_dict()) # or use this to copy parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) 
        # self.memory = PrioritizedReplayBuffer(cfg)
        self.memory = PrioritizedReplayBufferQue(cfg)
        self.update_flag = False 
        if is_share_agent:
            self.policy_net.share_memory()
            self.optimizer = SharedAdam(self.policy_net.parameters(), lr=cfg.lr)
            self.optimizer.share_memory()
        # ray中share_agent
        self.share_policy_ray = QNetwork(cfg).to(self.device)
        self.share_target_ray = QNetwork(cfg).to(self.device)
        self.optimizer_ray = SharedAdam(self.share_policy_ray.parameters(), lr=cfg.lr)

    def sample_action(self, state):
        ''' sample action with e-greedy policy
        '''
        self.sample_count += 1
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        else:
            action = random.randrange(self.n_actions)
        return action

    def predict_action(self,state):
        ''' predict action
        '''
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        return action
    def update(self, share_agent=None):
        if len(self.memory) < self.batch_size: # when transitions in memory donot meet a batch, not update
            # print ("self.batch_size = ", self.batch_size)
            return
        # sample a batch of transitions from replay buffer
        (state_batch, action_batch, reward_batch, next_state_batch, done_batch), idxs_batch, weights_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1) # shape(batchsize,1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1) # shape(batchsize,1)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1) # shape(batchsize,1)

        weights_batch = torch.tensor(weights_batch, device=self.device, dtype=torch.float).unsqueeze(1) # shape(batchsize,1)
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch) # shape(batchsize,1),requires_grad=True
        next_max_q_value_batch = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1) 
        expected_q_value_batch = reward_batch + self.gamma * next_max_q_value_batch* (1-done_batch)
        loss = torch.mean(torch.pow((q_value_batch - expected_q_value_batch) * weights_batch, 2))
        
        td_errors = torch.abs(q_value_batch - expected_q_value_batch).cpu().detach().numpy() # shape(batchsize,1)
        self.memory.update_priorities(idxs_batch, td_errors) # update priorities of sampled transitions
        # backpropagation
        if share_agent is not None:
            # Clear the gradient of the previous step of share_agent
            share_agent.optimizer.zero_grad()
            loss.backward()
            # clip to avoid gradient explosion
            for param in self.policy_net.parameters():  
                param.grad.data.clamp_(-1, 1)
            # Copy the gradient from policy_net of local_agnet to policy_net of share_agent
            for param, share_param in zip(self.policy_net.parameters(), share_agent.policy_net.parameters()):
                share_param._grad = param.grad
            share_agent.optimizer.step()
            self.policy_net.load_state_dict(share_agent.policy_net.state_dict())
            if self.sample_count % self.target_update == 0: # target net update, target_update means "C" in pseucodes
                self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.optimizer.zero_grad()  
            loss.backward()
            # clip to avoid gradient explosion
            for param in self.policy_net.parameters():  
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step() 

            if self.sample_count % self.target_update == 0: # target net update, target_update means "C" in pseucodes
                self.target_net.load_state_dict(self.policy_net.state_dict()) 

    def update_ray(self, share_policy_state_dict):
        """Update the share_agent parameters with ray"""
        if len(self.memory) < self.batch_size: # when transitions in memory donot meet a batch, not update
            return share_policy_state_dict
        # sample a batch of transitions from replay buffer
        (state_batch, action_batch, reward_batch, next_state_batch, done_batch), idxs_batch, weights_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1) # shape(batchsize,1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1) # shape(batchsize,1)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1) # shape(batchsize,1)

        weights_batch = torch.tensor(weights_batch, device=self.device, dtype=torch.float).unsqueeze(1) # shape(batchsize,1)
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch) # shape(batchsize,1),requires_grad=True
        next_max_q_value_batch = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1) 
        expected_q_value_batch = reward_batch + self.gamma * next_max_q_value_batch* (1-done_batch)
        loss = torch.mean(torch.pow((q_value_batch - expected_q_value_batch) * weights_batch, 2))
        
        td_errors = torch.abs(q_value_batch - expected_q_value_batch).cpu().detach().numpy() # shape(batchsize,1)
        self.memory.update_priorities(idxs_batch, td_errors) # update priorities of sampled transitions

        # 加载最新的共享智能体的网络参数
        self.share_policy_ray.load_state_dict(share_policy_state_dict)
        self.optimizer_ray.zero_grad()
        loss.backward()
        # clip to avoid gradient explosion
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        # 将local_agent计算出的梯度参数传给share_policy_ray
        for param, share_param in zip(self.policy_net.parameters(), self.share_policy_ray.parameters()):
            share_param._grad = param.grad
        # 更新share_policy_ray网络的参数
        self.optimizer_ray.step()
        # 将更新后的share_policy_ray网络的参数传给local_agent
        self.policy_net.load_state_dict(self.share_policy_ray.state_dict())
        if self.sample_count % self.target_update == 0: # target net update, target_update means "C" in pseucodes
            self.target_net.load_state_dict(self.policy_net.state_dict())  
        # 将更新后的share_policy_ray网络的参数传回ShareAgent类
        return self.share_policy_ray.state_dict()

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        checkpoint = torch.load(f"{fpath}/checkpoint.pt", map_location=self.device)
        self.target_net.load_state_dict(checkpoint)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


@ray.remote
class ShareAgent:
    def __init__(self, cfg):
        '''共享智能体类
        Args:
            cfg (class): 超参数类
        '''
        self.policy_net = QNetwork(cfg).to(cfg.device)
        self.target_net = QNetwork(cfg).to(cfg.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) 
        self.lr = cfg.lr
        # self.memory = ReplayBuffer(cfg.buffer_size)

    def get_parameters(self):
        return self.policy_net.state_dict()
    
    def receive_parameters(self, policy_net):
        self.policy_net.load_state_dict(policy_net)

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        checkpoint = torch.load(f"{fpath}/checkpoint.pt")
        self.target_net.load_state_dict(checkpoint)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)

    def update_parameters(self, local_net):
        """training algorithm in ShareAgent"""
        self.optimizer.zero_grad()
        for param, share_param in zip(local_net.parameters(), self.policy_net.parameters()):
            share_param._grad = param.grad
        self.optimizer.step()
        return self.policy_net
