#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: John
LastEditTime: 2023-03-29 13:06:23
@Discription: 
@Environment: python 3.7.7
'''
'''off-policy
'''

import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
import ray
import numpy as np
from common.layers import ValueNetwork
from common.memories import ReplayBuffer
from common.optms import SharedAdam


class Agent:
    def __init__(self,cfg, is_share_agent = False):
        '''智能体类
        Args:
            cfg (class): 超参数类
            is_share_agent (bool, optional): 是否为共享的 Agent ，多进程下使用，默认为 False
        '''
        self.n_actions = cfg.n_actions  
        self.device = torch.device(cfg.device) 
        self.gamma = cfg.gamma  
        ## e-greedy 策略相关参数
        self.sample_count = 0  # 采样动作计数
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update
        self.policy_net = ValueNetwork(cfg).to(self.device)
        # summary(self.policy_net, (1,4))
        self.target_net = ValueNetwork(cfg).to(self.device)
        ## copy parameters from policy net to target net
        # for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): 
        #     target_param.data.copy_(param.data)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # or use this to copy parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) 
        if is_share_agent:
            self.policy_net.share_memory()
            self.optimizer = SharedAdam(self.policy_net.parameters(), lr=cfg.lr)
            self.optimizer.share_memory()
            ## The multiprocess DQN algorithm does not use the target_net in share_agent
            # self.target_net.share_memory()
            # self.target_optimizer = SharedAdam(self.target_net.parameters(), lr=cfg.lr)
            # self.target_optimizer.share_memory()
        self.memory = ReplayBuffer(cfg.buffer_size)
        
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
            with torch.no_grad():
                state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        else:
            action = random.randrange(self.n_actions)
        return action
    # @torch.no_grad()
    # def sample_action(self, state):
    #     ''' sample action with e-greedy policy
    #     '''
    #     self.sample_count += 1
    #     # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
    #     self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
    #         math.exp(-1. * self.sample_count / self.epsilon_decay) 
    #     if random.random() > self.epsilon:
    #         state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
    #         q_values = self.policy_net(state)
    #         action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
    #     else:
    #         action = random.randrange(self.n_actions)
    #     return action
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
    def update(self, share_agent=None):
        if len(self.memory) < self.batch_size: # when transitions in memory donot meet a batch, not update
            return
        # sample a batch of transitions from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1) # shape(batchsize,1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1) # shape(batchsize,1)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1) # shape(batchsize,1)
        # print(state_batch.shape,action_batch.shape,reward_batch.shape,next_state_batch.shape,done_batch.shape)
        # compute current Q(s_t,a), it is 'y_j' in pseucodes
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch) # shape(batchsize,1),requires_grad=True
        # print(q_values.requires_grad)
        # compute max(Q(s_t+1,A_t+1)) respects to actions A, next_max_q_value comes from another net and is just regarded as constant for q update formula below, thus should detach to requires_grad=False
        next_max_q_value_batch = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1) 
        # print(q_values.shape,next_q_values.shape)
        # compute expected q value, for terminal state, done_batch[0]=1, and expected_q_value=rewardcorrespondingly
        expected_q_value_batch = reward_batch + self.gamma * next_max_q_value_batch* (1-done_batch)
        # print(expected_q_value_batch.shape,expected_q_value_batch.requires_grad)
        loss = nn.MSELoss()(q_value_batch, expected_q_value_batch)  # shape same to  
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
 
    def update_ray(self, share_agent_policy_net, share_agent_optimizer):
        """Update the share_agent parameters with ray"""
        batch_size = min(len(self.memory), self.batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1) # shape(batchsize,1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1) # shape(batchsize,1)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1) # shape(batchsize,1)
        # compute current Q(s_t,a), it is 'y_j' in pseucodes
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch) # shape(batchsize,1),requires_grad=True
        # compute max(Q(s_t+1,A_t+1)) respects to actions A, next_max_q_value comes from another net and is just regarded as constant for q update formula below, thus should detach to requires_grad=False
        next_max_q_value_batch = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1) 
        # compute expected q value, for terminal state, done_batch[0]=1, and expected_q_value=rewardcorrespondingly
        expected_q_value_batch = reward_batch + self.gamma * next_max_q_value_batch* (1-done_batch)
        loss = nn.MSELoss()(q_value_batch, expected_q_value_batch)
        share_agent_optimizer.zero_grad()
        loss.backward()
        # clip to avoid gradient explosion
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        for param, share_param in zip(self.policy_net.parameters(), share_agent_policy_net.parameters()):
            share_param._grad = param.grad
        share_agent_optimizer.step()
        self.policy_net.load_state_dict(share_agent_policy_net.state_dict())
        if self.sample_count % self.target_update == 0: # target net update, target_update means "C" in pseucodes
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return share_agent_policy_net, share_agent_optimizer

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        self.target_net.load_state_dict(torch.load(f"{fpath}/checkpoint.pt"))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


@ray.remote
class ShareAgent:
    def __init__(self,cfg):
        self.policy_net = ValueNetwork(cfg).to(cfg.device)
        self.target_net = ValueNetwork(cfg).to(cfg.device)
        # print(f'self.policy_net:{self.policy_net}')
        self.optimizer = SharedAdam(self.policy_net.parameters(), lr=cfg.lr)
        # self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) 
        self.lr = cfg.lr
        # self.memory = ReplayBuffer(cfg.buffer_size)

    def get_parameters(self):
        return self.policy_net, self.optimizer
    
    def receive_parameters(self, policy_net, optimizer):
        # self.policy_net = policy_net
        # self.optimizer = optimizer
        # 
        self.policy_net.load_state_dict(policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr) 

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        self.policy_net.load_state_dict(torch.load(f"{fpath}/checkpoint.pt"))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

    
    def update_parameters(self, local_net):
        """training algorithm in ShareAgent"""
        self.optimizer.zero_grad()
        for param, share_param in zip(local_net.parameters(), self.policy_net.parameters()):
            share_param._grad = param.grad
        self.optimizer.step()
        return self.policy_net
    
    def get_share_net(self):
        return self.policy_net