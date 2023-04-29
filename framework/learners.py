#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-27 23:45:49
LastEditor: JiangJi
LastEditTime: 2023-04-29 10:41:47
Discription: 
'''
import ray 
import asyncio
@ray.remote(num_cpus=1)
class Learner:
    def __init__(self, cfg, id = None, policy = None ) -> None:
        self.cfg = cfg
        self.id = id
        self.policy = policy
    async def run(self, data_server):
        
        while not await data_server.check_episode_limit.remote():
            # print(f"Learner is running {await data_server.get_episode.remote()}")
            training_data = await data_server.dequeue_training_data.remote()
            self.policy.update(**training_data)
            policy_params = self.policy.get_params()
            # print("learner policy_params",policy_params['policy_net.layers.2.0.weight'][0][:10])
            await data_server.enqueue_policy_params.remote(policy_params)
            
