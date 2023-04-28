#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-27 23:45:49
LastEditor: JiangJi
LastEditTime: 2023-04-28 16:55:52
Discription: 
'''
import ray 
@ray.remote
class Learner:
    def __init__(self, cfg, id = None, policy = None, data_processor = None) -> None:
        self.cfg = cfg
        self.id = id
        self.policy = policy
    async def run(self, data_server):
        while not await data_server.check_episode_limit.remote():
            training_data = await data_server.dequeue_training_data.remote()
            self.policy.update(training_data)
            policy_params = self.policy.get_policy_params()
            await data_server.enqueue_policy_params.remote(policy_params)
