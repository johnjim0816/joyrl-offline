#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: John
LastEditTime: 2023-04-19 02:09:29
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
from common.optms import SharedAdam
from algos.base.buffers import BufferCreator
from algos.base.agents import BaseAgent
from algos.base.networks import ValueNetwork

class Agent(BaseAgent):
    def __init__(self, cfg, is_share_agent = False):
        super(Agent, self).__init__()
        '''智能体类
        Args:
            cfg (class): 超参数类
            is_share_agent (bool, optional): 是否为共享的 Agent ，多进程下使用，默认为 False
        '''
        self.cfg = cfg
        self.obs_space = cfg.obs_space
        self.action_space = cfg.action_space
        self.device = torch.device(cfg.general_cfg.device) 
        self.gamma = cfg.algo_cfg.gamma  
        ## e-greedy 策略相关参数
        self.sample_count = 0  # 采样动作计数
        self.epsilon_start = cfg.algo_cfg.epsilon_start
        self.epsilon_end = cfg.algo_cfg.epsilon_end
        self.epsilon_decay = cfg.algo_cfg.epsilon_decay
        self.batch_size = cfg.algo_cfg.batch_size
        self.target_update = cfg.algo_cfg.target_update
        self.is_share_agent = is_share_agent
        self.memory = BufferCreator(cfg.algo_cfg)()
        self.create_graph()
    def create_graph(self):
        self.input_size = [None, self.obs_space.shape[0]]
        action_dim = self.action_space.n
        self.policy_net = ValueNetwork(self.cfg.algo_cfg, self.input_size, action_dim)
        self.target_net = ValueNetwork(self.cfg.algo_cfg, self.input_size, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # or use this to copy parameters
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.cfg.algo_cfg.lr) 
        if self.is_share_agent:
            self.policy_net.share_memory()
            self.optimizer = SharedAdam(self.policy_net.parameters(), lr=self.algo_cfg.lr)
            self.optimizer.share_memory()
            ## The multiprocess DQN algorithm does not use the target_net in share_agent
            # self.target_net.share_memory()
            # self.target_optimizer = SharedAdam(self.target_net.parameters(), lr=cfg.lr)
            # self.target_optimizer.share_memory()
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
    def update(self, share_agent=None):
        # 从 replay buffer 中采样
        exps = self.memory.sample(self.batch_size)
        # 将采样的数据转换为 tensor
        states = torch.tensor(np.array([exp.state for exp in exps]), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.array([exp.action for exp in exps]), device=self.device, dtype=torch.long).unsqueeze(dim=1)
        rewards = torch.tensor(np.array([exp.reward for exp in exps]), device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        next_states = torch.tensor(np.array([exp.next_state for exp in exps]), device=self.device, dtype=torch.float32)
        dones = torch.tensor(np.array([exp.done for exp in exps]), device=self.device, dtype=torch.float32).unsqueeze(dim=1)

        # 计算当前状态的 Q 值
        q_values = self.policy_net(states).gather(1, actions)
        # 计算下一个状态的最大 Q 值
        next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(dim=1)
        # 计算目标 Q 值
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        # 计算损失
        loss = nn.MSELoss()(q_values, target_q_values)
        if share_agent is not None: # 多进程下使用
            share_agent.optimizer.zero_grad()
            loss.backward()
            # clip 防止梯度爆炸
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            # 复制梯度到共享的 policy_net
            for param, share_param in zip(self.policy_net.parameters(), share_agent.policy_net.parameters()):
                share_param._grad = param.grad
            share_agent.optimizer.step()
            self.policy_net.load_state_dict(share_agent.policy_net.state_dict())

            if self.sample_count % self.target_update == 0: # 每 C 步更新一次 target_net
                self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.optimizer.zero_grad()
            loss.backward()
            # clip 防止梯度爆炸
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            if self.sample_count % self.target_update == 0: # 每 C 步更新一次 target_net
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
        # 创建文件夹
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        self.policy_net.load_state_dict(torch.load(f"{fpath}/checkpoint.pt"))



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