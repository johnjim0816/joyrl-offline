#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 16:16:04
LastEditor: JiangJi
LastEditTime: 2023-05-08 00:22:50
Discription: 
'''
import ray
from ray.util.queue import Queue, Empty, Full

@ray.remote
class DataServer:
    def __init__(self, cfg) -> None:
        self.curr_episode = 0
        self.sample_count = 0
        self.update_step = 0
        self.max_epsiode = cfg.max_epsiode
        self.best_reward = -float('inf')
    def increase_episode(self):
        self.curr_episode += 1
    def check_episode_limit(self):
        return self.curr_episode >= self.max_epsiode
    def get_episode(self):
        return self.curr_episode
    def increase_sample_count(self):
        self.sample_count += 1
    def get_sample_count(self):
        return self.sample_count
    def increase_update_step(self):
        self.update_step += 1
    def get_update_step(self):
        return self.update_step

