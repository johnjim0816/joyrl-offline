#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2023-05-19 00:47:44
Discription: use defaultdict to define Q table
Environment: 
'''
import numpy as np
import math
import torch
from collections import defaultdict

class Agent(object):
    def __init__(self,cfg):
        '''智能体类
        Args:
            cfg (class): 超参数类
        '''
        self.n_actions = cfg.n_actions 
        self.exploration_type = 'e-greedy' # 探索策略如 e-greedy ，boltzmann ，softmax， ucb 等
           
        self.epsilon = cfg.epsilon_start
        self.sample_count = 0  
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.Q_table  = defaultdict(lambda: np.zeros(self.n_actions)) # 使用嵌套字典来表示 Q(s,a)，并将指定所有的 Q_table 创建时， Q(s,a) 初始设置为 0
    def sample_action(self, state):
        ''' 以 e-greedy 策略训练时选择动作 
        Args:
            state (array): 状态
        Returns:
            action (int): 动作
        ''' 
        if self.exploration_type == 'e-greedy':
            action = self._epsilon_greedy_sample_action(state)
        else:
            raise NotImplementedError
        return action
    def predict_action(self,state):
        ''' 预测动作
        Args:
            state (array): 状态
        Returns:
            action (int): 动作
        '''
        if self.exploration_type == 'e-greedy':
            action = self._epsilon_greedy_predict_action(state)
        else:
            raise NotImplementedError
        return action
    def _epsilon_greedy_sample_action(self, state):
        ''' 
        采用 epsilon-greedy 策略进行动作选择 
        Args: 
            state (array): 状态
        Returns: 
            action (int): 动作
        ''' 
        self.sample_count += 1
        # epsilon 值需要衰减，衰减方式可以是线性、指数等，以平衡探索和开发
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if np.random.uniform(0, 1) > self.epsilon:
            action = np.argmax(self.Q_table[str(state)]) # 选择具有最大 Q 值的动作
        else:
            action = np.random.choice(self.n_actions) # 随机选择一个动作
        return action
    def _epsilon_greedy_predict_action(self,state):
        ''' 
        使用 epsilon-greedy 算法进行动作预测 
        Args: 
            state (array): 状态
        Returns: 
            action (int): 动作 
        ''' 
        action = np.argmax(self.Q_table[str(state)])
        return action
    def update(self, state, action, reward, next_state, done):
        ''' 更新Q(s,a)
        Args:
            state (array): 当前状态 
            action (int): 当前动作 
            reward (float): 当前奖励信号 
            next_state (array): 下一个状态 
            done (bool): 表示是否达到终止状态 
        '''
        Q_predict = self.Q_table[str(state)][action] 
        if done: # 终止状态 
            Q_target = reward  
        else:
            Q_target = reward + self.gamma * np.max(self.Q_table[str(next_state)]) 
        self.Q_table[str(state)][action] += self.lr * (Q_target - Q_predict)
    def save_model(self,path):
        '''
        保存模型
        Args:
            path (str): 模型存储路径 
        '''
        import dill
        from pathlib import Path
        # 确保存储路径存在 
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(
            obj=self.Q_table,
            f=path+"Qleaning_model.pkl",
            pickle_module=dill
        )
        print("Model saved!")
    def load_model(self, path):
        '''
        根据模型路径导入模型
        Args:
            fpath (str): 模型路径
        '''
        import dill
        self.Q_table =torch.load(f=path+'Qleaning_model.pkl',pickle_module=dill)
        print("Mode loaded!")