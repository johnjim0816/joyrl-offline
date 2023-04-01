#!/usr/bin/env python
# coding=utf-8
'''
Author: GuoShiCheng
Email: guoshichenng@gmail.com
Date: 2023-4-01 15:00:40
LastEditor: GuoShiCheng
LastEditTime: 2023-4-01 15:00:40
Discription: 
Environment: python 3.7.7
'''

import ray

@ray.remote
class GlobalVarActor:
    """
    Global Variables
    """
    def __init__(self):
        self.episode = 0
        self.best_reward = 0.

    def add_episode(self, episode=1):
        self.episode += episode

    def read_episode(self):
        return self.episode
    
    def add_read_episode(self, episode=1):
        # 
        self.episode += episode
        return self.episode
    
    def set_best_reward(self, mean_eval_reward):
        self.best_reward = mean_eval_reward
        return self.best_reward
    
    def read_best_reward(self):
        return self.best_reward