#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 16:16:04
LastEditor: JiangJi
LastEditTime: 2023-04-28 16:45:51
Discription: 
'''
import ray
import asyncio

@ray.remote
class DataServer:
    def __init__(self, cfg) -> None:
        self.curr_episode = 0
        self.max_epsiode = cfg.max_epsiode
        self.exp_queue = asyncio.Queue()
        self.policy_params_queue = asyncio.Queue()
        self.training_data = []
    def increment_episode(self):
        self.curr_episode += 1
    def check_episode_limit(self):
        return self.curr_episode >= self.max_epsiode
    def get_episode(self):
        return self.curr_episode
    async def enqueue_exp(self, exp):
        await self.exp_queue.put(exp)
    async def dequeue_exp(self):
        return await self.exp_queue.get()
    async def euqueue_policy_params(self, policy_params):
        await self.policy_params_queue.put(policy_params)
    async def dequeue_policy_params(self):
        return await self.policy_params_queue.get()
    async def enqueue_training_data(self, data):
        self.training_data.append(data)
    async def dequeue_training_data(self):
        return self.training_data.put()