#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-24 23:26:57
LastEditor: JiangJi
LastEditTime: 2023-04-26 22:57:55
Discription: 
'''
import ray

@ray.remote
class GlobalVarRecorder:
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

@ray.remote
class StatsRecorder:
    def __init__(self,cfg) -> None:
        pass