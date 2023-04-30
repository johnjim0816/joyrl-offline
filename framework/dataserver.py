#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 16:16:04
LastEditor: JiangJi
LastEditTime: 2023-04-30 01:42:36
Discription: 
'''
import ray
import asyncio

@ray.remote
class DataServer:
    def __init__(self, cfg) -> None:
        self.curr_episode = 0
        self.max_epsiode = cfg.max_epsiode
        self.exp_queue = asyncio.Queue() # maxsize得调整，不然卡住
        self.policy_params_queue = asyncio.Queue()
        self.training_data = asyncio.Queue()
        self.best_reward = -float('inf')
        self.policy_params = None
    def increment_episode(self):
        self.curr_episode += 1
    def check_episode_limit(self):
        return self.curr_episode >= self.max_epsiode
    def get_episode(self):
        return self.curr_episode
    def set_policy_params(self, policy_params):
        self.policy_params = policy_params
    def get_policy_params(self):
        return self.policy_params
    def set_best_reward(self, reward):
        self.best_reward = reward
    def get_best_reward(self):
        return self.best_reward
    async def enqueue_exp(self, exp):
        await self.exp_queue.put(exp)
    async def dequeue_exp(self):
        return await self.exp_queue.get()
    async def enqueue_policy_params(self, policy_params):
        self.set_policy_params(policy_params)
        await self.policy_params_queue.put(policy_params)
    async def dequeue_policy_params(self):
        # return self.policy_params
        if self.policy_params_queue.empty():
            return self.policy_params
        return await self.policy_params_queue.get()
    async def enqueue_training_data(self, data):
        await self.training_data.put(data)
    async def dequeue_training_data(self):
        return await self.training_data.get()