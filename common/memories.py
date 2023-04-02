#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-10 15:27:16
@LastEditor: John
LastEditTime: 2023-03-31 23:39:29
@Discription: 
@Environment: python 3.7.7
'''
import random
import numpy as np
from collections import deque
import operator
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # 经验回放的容量
        self.buffer = [] # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        state, action, reward, next_state, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)

class ReplayBufferQue:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer = deque(maxlen=self.capacity)
    def push(self,transitions):
        '''_summary_
        Args:
            trainsitions (tuple): _description_
        '''
        self.buffer.append(transitions)
    def sample(self, batch_size: int, sequential: bool = False):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if sequential: # sequential sampling
            rand = random.randint(0, len(self.buffer) - batch_size)
            batch = [self.buffer[i] for i in range(rand, rand + batch_size)]
            return zip(*batch)
        else:
            batch = random.sample(self.buffer, batch_size)
            return zip(*batch)
    def clear(self):
        self.buffer.clear()
    def __len__(self):
        return len(self.buffer)

class PGReplay(ReplayBufferQue):
    '''replay buffer for policy gradient based methods, each time these methods will sample all transitions
    Args:
        ReplayBufferQue (_type_): _description_
    '''
    def __init__(self):
        self.buffer = deque()
    def sample(self):
        ''' sample all the transitions
        '''
        batch = list(self.buffer)
        return zip(*batch)
    
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object) # 存储样本
        self.write_idx = 0 # 写入样本的索引
        self.count = 0 # 当前存储的样本数量
    
    def add(self, priority, exps):
        ''' 添加一个样本到叶子节点，并更新其父节点的优先级
        '''
        idx = self.write_idx + self.capacity - 1 # 样本的索引
        self.data[self.write_idx] = exps # 写入样本
        self.update(idx, priority) # 更新样本的优先级
        self.write_idx = (self.write_idx + 1) % self.capacity # 更新写入样本的索引
        if self.count < self.capacity:
            self.count += 1
    
    def update(self, idx, priority):
        ''' 更新叶子节点的优先级，并更新其父节点的优先级
        Args:
            idx (int): 样本的索引
            priority (float): 样本的优先级
        '''
        diff = priority - self.tree[idx] # 优先级的差值
        self.tree[idx] = priority
        while idx != 0: 
            idx = (idx - 1) // 2
            self.tree[idx] += diff
    
    def get_leaf(self, v):
        ''' 根据优先级的值采样对应区间的叶子节点样本
        '''
        idx = 0
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                break
            if v <= self.tree[left]:
                idx = left
            else:
                v -= self.tree[left]
                idx = right
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]
    def get_data(self, indices):
        return [self.data[idx - self.capacity + 1] for idx in indices]
    
    def total(self):
        ''' 返回所有样本的优先级之和，即根节点的值
        '''
        return self.tree[0]
    
    def max_prior(self):
        ''' 返回所有样本的最大优先级
        '''
        return np.max(self.tree[self.capacity-1:self.capacity+self.write_idx-1])
    
class PrioritizedReplayBuffer:
    def __init__(self, cfg):
        self.capacity = cfg.buffer_size
        self.alpha = cfg.per_alpha # 优先级的指数参数，越大越重要，越小越不重要
        self.epsilon = cfg.per_epsilon # 优先级的最小值，防止优先级为0
        self.beta = cfg.per_beta # importance sampling的参数
        self.beta_annealing = cfg.per_beta_annealing # beta的增长率
        self.tree = SumTree(self.capacity)
        self.max_priority = 1.0
    
    def push(self, exps):
        ''' 添加样本
        '''
        priority = self.max_priority if self.tree.total() == 0 else self.tree.max_prior()
        self.tree.add(priority, exps)
    
    def sample(self, batch_size):
        ''' 采样一个批量样本
        '''
        indices = [] # 采样的索引
        priorities = [] # 采样的优先级
        exps = [] # 采样的样本
        segment = self.tree.total() / batch_size
        self.beta = min(1.0, self.beta  + self.beta_annealing)
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            p = np.random.uniform(a, b) # 采样一个优先级
            idx, priority, exp = self.tree.get_leaf(p) # 采样一个样本
            indices.append(idx)
            priorities.append(priority)
            exps.append(exp)
        # 重要性采样, weight = (N * P(i)) ^ (-beta) / max_weight
        sample_probs = np.array(priorities) / self.tree.total()
        weights = (self.tree.count * sample_probs) ** (-self.beta) # importance sampling的权重
        weights /= weights.max() # 归一化
        indices = np.array(indices)
        return zip(*exps), indices, weights
    
    def update_priorities(self, indices, priorities):
        ''' 更新样本的优先级
        '''
        priorities = np.abs(priorities) # 取绝对值
        for idx, priority in zip(indices, priorities):
            # 控制衰减的速度, priority = (priority + epsilon) ^ alpha
            priority = (priority + self.epsilon) ** self.alpha
            priority = np.minimum(priority, self.max_priority)
            self.tree.update(idx, priority)
    def __len__(self):
        return self.tree.count

class PrioritizedReplayBufferQue:
    def __init__(self, cfg):
        self.capacity = cfg.buffer_size
        self.alpha = cfg.per_alpha # 优先级的指数参数，越大越重要，越小越不重要
        self.epsilon = cfg.per_epsilon # 优先级的最小值，防止优先级为0
        self.beta = cfg.per_beta # importance sampling的参数
        self.beta_annealing = cfg.per_beta_annealing # beta的增长率
        self.buffer = deque(maxlen=self.capacity)
        self.priorities = deque(maxlen=self.capacity)
        self.count = 0 # 当前存储的样本数量
        self.max_priority = 1.0
    def push(self,exps):
        self.buffer.append(exps)
        self.priorities.append(max(self.priorities, default=self.max_priority))
        self.count += 1
    def sample(self, batch_size):
        priorities = np.array(self.priorities)
        probs = priorities/sum(priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        weights = (self.count*probs[indices])**(-self.beta)
        weights /= weights.max()
        exps = [self.buffer[i] for i in indices]
        return zip(*exps), indices, weights
    def update_priorities(self, indices, priorities):
        priorities = np.abs(priorities)
        priorities = (priorities + self.epsilon) ** self.alpha
        priorities = np.minimum(priorities, self.max_priority).flatten()
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    def __len__(self):
        return self.count