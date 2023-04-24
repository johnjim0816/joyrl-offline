#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-03-10 23:35:32
LastEditor: JiangJi
LastEditTime: 2023-03-11 00:13:55
Discription: 
'''
from collections import defaultdict
import numpy as np
import math
import torch

class Agent(object):
    def __init__(self,cfg):
        '''智能体类
        Args:
            cfg (class): 超参数类
        '''
        self.n_actions = cfg.n_actions 
        self.lr = cfg.lr 
        self.gamma = cfg.gamma    
        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table  = defaultdict(lambda: np.zeros(self.n_actions)) # 使用嵌套字典来表示 Q(s,a)，并将指定所有的 Q_table 创建时， Q(s,a) 初始设置为 0
        self.sample_count = 0  
    def sample_action(self, state):
        ''' 另一种实现 e-greedy 策略的方式
        Args:
            state (array): 状态
        Returns:
            action (int): 动作
        '''
        self.sample_count += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) # 选择随机动作的概率 epsilon 是对数衰减的
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)]) # 选取与最大 q 值相对应的动作
        else:
            action = np.random.choice(self.n_actions) # 随机选取动作
        return action
    def predict_action(self,state):
        ''' 预测动作
        Args:
            state (array): 状态
        Returns:
            action (int): 动作
        '''
        action = np.argmax(self.Q_table[str(state)])
        return action
    def update(self, state, action, reward, next_state, next_action, done):
        ''' 更新模型
        Args:
            state (array): 当前状态 
            action (int): 当前动作 
            reward (float): 当前奖励信号 
            next_state (array): 下一个状态 
            done (bool): 表示是否达到终止状态 
        '''
        Q_predict = self.Q_table[str(state)][action]
        if done:
            Q_target = reward  # 终止状态 
        else:
            Q_target = reward + self.gamma * self.Q_table[str(next_state)][next_action] # 与 Q learning 的唯一区别
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict) 
    def save_model(self,fpath):
        '''
        保存模型
        Args:
            path (str): 模型存储路径 
        '''
        import dill
        from pathlib import Path
        ## 确保存储路径存在 
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(
            obj=self.Q_table,
            f=f"{fpath}/checkpoint.pkl",
            pickle_module=dill
        )
    def load_model(self, fpath):
        '''
        根据模型路径导入模型
        Args:
            fpath (str): 模型路径
        '''
        import dill
        self.Q_table=torch.load(f=f"{fpath}/checkpoint.pkl",pickle_module=dill)