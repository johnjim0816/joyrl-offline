#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 17:10:34
LastEditor: JiangJi
LastEditTime: 2023-04-29 00:22:29
Discription: 
'''
import ray
import asyncio

@ray.remote

class Collector:
    def __init__(self, cfg, data_handler=None) -> None:
        self.cfg = cfg
        self.data_handler = data_handler
    async def run(self, data_server):
        print("Collector is running")
        while not await data_server.check_episode_limit.remote():
            exp = await data_server.dequeue_exp.remote()
            self.data_handler.add_transition(exp)
            training_data = self.data_handler.sample_training_data()
            # print("training_data",training_data)
            if training_data:
                await data_server.enqueue_training_data.remote(training_data)
