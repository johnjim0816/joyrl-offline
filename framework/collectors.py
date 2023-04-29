#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 17:10:34
LastEditor: JiangJi
LastEditTime: 2023-04-29 10:41:22
Discription: 
'''
import ray
import asyncio

@ray.remote(num_cpus=1)
class Collector:
    def __init__(self, cfg, data_handler=None) -> None:
        self.cfg = cfg
        self.data_handler = data_handler
    async def run(self, data_server):
        while not await data_server.check_episode_limit.remote():
            # print(f"Collector is running {await data_server.get_episode.remote()}")
            # print("data_handler",len(self.data_handler.buffer))
            exp = await data_server.dequeue_exp.remote()
            self.data_handler.add_transition(exp)
            training_data = self.data_handler.sample_training_data()
            # print("training_data",training_data)
            if training_data:
                await data_server.enqueue_training_data.remote(training_data)
            