import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical,Normal
import torch.utils.data as Data
import numpy as np
import pickle

from common.models import ActorSoftmax, ActorNormal, Critic
from common.memories import PGReplay

class Agent:
    def __init__(self,cfg) -> None:
        '''智能体类
        Args:
            cfg (class): 超参数类
        '''
        self.device = torch.device(cfg.device)
        self.continuous = cfg.continuous # 连续动作空间
        self.action_space = cfg.action_space
        if self.continuous:
            self.action_scale = torch.tensor((self.action_space.high - self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0) # 动作空间放缩
            self.action_bias = torch.tensor((self.action_space.high + self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0) # 动作空间平移
            self.policynet = ActorNormal(cfg.n_states,cfg.n_actions, hidden_dim = cfg.actor_hidden_dim).to(self.device) # 定义连续动作的策略网络
        else:
            self.policynet = ActorSoftmax(cfg.n_states,cfg.n_actions, hidden_dim = cfg.actor_hidden_dim).to(self.device) # 定义离散动作的策略网络
        self.actor_optimizer = torch.optim.Adam(self.policynet.parameters(), lr=cfg.lr) # 定义优化器

        self.expert_path = f"{cfg.expert_path}" # 专家数据路径
        with open(self.expert_path, 'rb') as f:
            expert_data = pickle.load(f)
        self.expert_states = np.array(expert_data['states']) ; self.expert_actions = np.array(expert_data['actions']) #得到专家状态动作对


    @torch.no_grad()
    def predict_action(self,state):
        '''预测动作
        Args:
            state (array): 状态
        Returns:
            action (float|int): 连续动作|离散动作
        '''
        if self.continuous:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            mu, sigma = self.policynet(state) # 根据状态 state 推理得到对应动作向量的均值和方差
            mean = mu * self.action_scale + self.action_bias # 对均值进行放缩和平移
            std = sigma
            dist = Normal(mean, std) # 定义一个均值为 mean ，方差为 std 的高斯分布
            action = dist.sample() # 从高斯分布中采样一个动作向量
            self.log_probs = dist.log_prob(action).detach() # 计算样本在分布中概率密度的对数
            return action.detach().cpu().numpy()[0]
        else: 
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            probs = self.policynet(state) # 得到策略网络推理的动作概率分布向量
            dist = Categorical(probs) # 定义一个 Categorical 分布
            action = dist.sample() # 从分布中采样一个样本
            self.log_probs = dist.log_prob(action).detach() # 计算样本的对数概率
            return action.detach().cpu().numpy().item()
        
    def update(self, expert_states, expert_actions):
        '''离散动作环境更新策略
        Args:
            expert_states (array): 专家状态
            expert_actions (int): 专家动作
        '''
        expert_states = torch.tensor(expert_states, dtype=torch.float).to(self.device) # 专家状态
        expert_actions = torch.tensor(expert_actions).view(-1, 1).to(self.device) # 专家动作
        # 使用最大似然估计
        probs = self.policynet(expert_states).gather(1, expert_actions) # 策略网络预测动作概率，并从中选择与专家一致的动作的动作概率
        log_probs = torch.log(probs) # 取概率的对数
        actor_loss = torch.mean(-log_probs)  # 动作概率对数取负求平均作为损失
        self.actor_optimizer.zero_grad() # 将策略网络梯度清零
        actor_loss.backward() # 自动求导计算损失对于策略网络参数的梯度
        self.actor_optimizer.step() # 更新策略网络的参数

    def save_model(self, fpath):
        '''保存网络权重
        Args:
            fpath (str): 保存权重文件路径
        '''
        from pathlib import Path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.policynet.state_dict(), f"{fpath}/model.pth")

    def load_model(self, fpath):
        '''加载网络权重
        Args:
            fpath (str): 加载权重文件路径
        '''
        actor_ckpt = torch.load(f"{fpath}/model.pth", map_location=self.device)
        self.policynet.load_state_dict(actor_ckpt)
