#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-27 23:45:49
LastEditor: JiangJi
LastEditTime: 2023-04-29 00:20:28
Discription: 
'''
import ray 
@ray.remote
class Learner:
    def __init__(self, cfg, id = None, policy = None ) -> None:
        self.cfg = cfg
        self.id = id
        self.policy = policy
    async def run(self, data_server):
        print("Learner is running")
        while not await data_server.check_episode_limit.remote():
            training_data = await data_server.dequeue_training_data.remote()
            self.policy.update(**training_data)
            policy_params = self.policy.get_params()
            await data_server.enqueue_policy_params.remote(policy_params)
