#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-12 00:50:49
@LastEditor: John
LastEditTime: 2023-03-24 22:53:55
@Discription: 
@Environment: python 3.7.7
'''
'''off-policy
'''

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
import ray
import numpy as np
from common.memories import ReplayBufferQue
from common.layers import ValueNetwork
from common.optms import SharedAdam


class Agent:
    def __init__(self, cfg, is_share_agent=False):
        '''智能体类
        Args:
            cfg (class): 超参数类
            is_share_agent (bool, optional): 是否为共享的 Agent ，多进程下使用，默认为 False
        '''
        self.n_actions = cfg.n_actions
        self.device = torch.device(cfg.device)
        self.gamma = cfg.gamma  # 折扣因子
        ## e-greedy parameters
        self.sample_count = 0  # sample count for epsilon decay
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update  # 目标网络更新频率
        self.policy_net = ValueNetwork(cfg).to(self.device)  # Q网络
        # summary(self.policy_net, (1,4))
        self.target_net = ValueNetwork(cfg).to(self.device)  # 目标网络
        # target_net copy from policy_net
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)
        # self.target_net.eval()  # donnot use BatchNormalization or Dropout
        # the difference between parameters() and state_dict() is that parameters() require_grad=True
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)  # 使用Adam优化器
        self.memory = ReplayBufferQue(cfg.buffer_size)  # 经验回放池
        self.update_flag = False
        if is_share_agent:
            self.policy_net.share_memory()
            self.optimizer = SharedAdam(self.policy_net.parameters(), lr=cfg.lr)
            self.optimizer.share_memory()

    def sample_action(self, state):
        '''采样动作
        Args:
            state(array): 状态
        Returns:
            action(int): 动作
        '''
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * math.exp(
            -1. * self.sample_count / self.epsilon_decay)
        if random.random() > self.epsilon:  # 使用 epsilon greedy
            action = self.predict_action(state)
        else:
            action = random.randrange(self.n_actions)
        return action

    def predict_action(self, state):
        ''' 预测动作
        Args:
            state(array): 状态
        Returns:
            actions(int): 动作
        '''
        with torch.no_grad():
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
            q_value = self.policy_net(state)
            action = q_value.max(1)[1].item()
        return action

    def update(self, share_agent=None):
        ''' 更新网络参数
        Args:
            share_agent: 是否为共享的Agent，多进程下使用，默认不共享
        '''
        if len(self.memory) < self.batch_size:  # when transitions in memory donot meet a batch, not update
            return
        # sample a batch of transitions from replay buffer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(self.batch_size)
        # convert to tensor
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  # shape(batchsize,1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(
            1)  # shape(batchsize,1)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1)  # shape(batchsize,1)
        # 计算当前Q(s_t, a_t)，即Q网络的输出，这里的gather函数的作用是根据action_batch中的值，从Q网络的输出中选出对应的Q值
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch)  # shape(batchsize,1)
        # Q网络计算 Q(s_t+1, a)
        next_q_value_batch = self.policy_net(next_state_batch)
        # 目标网络计算 Q'(s_t+1, a)，也是与DQN不同的地方
        next_target_value_batch = self.target_net(next_state_batch)
        # 计算 Q'(s_t+1, a=argmax Q(s_t+1, a))
        next_target_q_value_batch = next_target_value_batch.gather(1, torch.max(next_q_value_batch, 1)[1].unsqueeze(
            1))  # shape(batchsize,1)

        expected_q_value_batch = reward_batch + self.gamma * next_target_q_value_batch * (1 - done_batch)  # TD误差目标
        loss = nn.MSELoss()(q_value_batch, expected_q_value_batch)  # 均方误差损失函数
        if share_agent is not None:
            # Clear the gradient of the previous step of share_agent
            share_agent.optimizer.zero_grad()  # Pytorch 默认梯度会累计，这里需要显式将梯度设置为0
            loss.backward()  # 反向传播更新参数
            # clip to avoid gradient explosion
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            # Copy the gradient from policy_net of local_agnet to policy_net of share_agent
            for param, share_param in zip(self.policy_net.parameters(), share_agent.policy_net.parameters()):
                share_param._grad = param.grad
            share_agent.optimizer.step()
            self.policy_net.load_state_dict(share_agent.policy_net.state_dict())
            # 定期复制Q网络更新目标网络参数
            if self.sample_count % self.target_update == 0:  # target net update, target_update means "C" in pseucodes
                self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.optimizer.zero_grad()  # 梯度设置为0
            loss.backward()  # 反向传播更新参数
            # clip to avoid gradient explosion
            for param in self.policy_net.parameters():
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step()
            # 定期复制Q网络更新目标网络参数
            if self.sample_count % self.target_update == 0:  # target net update, target_update means "C" in pseucodes
                self.target_net.load_state_dict(self.policy_net.state_dict())

    def update_ray(self, share_agent_policy_net, share_agent_optimizer):
        """Update the share_agent parameters with ray"""
        batch_size = min(len(self.memory), self.batch_size)
        if len(self.memory) < self.batch_size:  # when transitions in memory donot meet a batch, not update
            return share_agent_policy_net, share_agent_optimizer
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device,
                                   dtype=torch.float)  # shape(batchsize,n_states)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)  # shape(batchsize,1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(
            1)  # shape(batchsize,1)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device,
                                        dtype=torch.float)  # shape(batchsize,n_states)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1)  # shape(batchsize,1)
        # 计算当前Q(s_t, a_t)，即Q网络的输出，这里的gather函数的作用是根据action_batch中的值，从Q网络的输出中选出对应的Q值
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch)  # shape(batchsize,1)
        # 计算 Q(s_t+1, a)
        next_q_value_batch = self.policy_net(next_state_batch)
        # 计算 Q'(s_t+1, a)，也是与DQN不同的地方
        next_target_value_batch = self.target_net(next_state_batch)
        # 计算 Q'(s_t+1, a=argmax Q(s_t+1, a))
        next_target_q_value_batch = next_target_value_batch.gather(1, torch.max(next_q_value_batch, 1)[1].unsqueeze(
            1))  # shape(batchsize,1)
        expected_q_value_batch = reward_batch + self.gamma * next_target_q_value_batch * (1 - done_batch)
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
        # 目标网络参数不频繁更新，而是定期从Q网络复制过来，这样有助于提升训练的稳定性和收敛性
        if self.sample_count % self.target_update == 0:  # target net update, target_update means "C" in pseucodes
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return share_agent_policy_net, share_agent_optimizer

    def save_model(self, path):
        '''保存模型
        Args:
            path(str): 存储路径
        '''
        from pathlib import Path
        # create path
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{path}/checkpoint.pt")

    def load_model(self, path):
        '''加载模型
        Args:
            path(str): 加载路径
        '''
        self.target_net.load_state_dict(torch.load(f"{path}/checkpoint.pt"))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


@ray.remote
class ShareAgent:
    def __init__(self, cfg):
        '''共享智能体类
        Args:
            cfg (class): 超参数类
        '''
        self.policy_net = ValueNetwork(cfg).to(cfg.device)
        self.target_net = ValueNetwork(cfg).to(cfg.device)
        self.optimizer = SharedAdam(self.policy_net.parameters(), lr=cfg.lr)
        self.lr = cfg.lr
        # self.memory = ReplayBuffer(cfg.buffer_size)

    def get_parameters(self):
        return self.policy_net, self.optimizer

    def receive_parameters(self, policy_net, optimizer):
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
