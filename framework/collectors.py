#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 17:10:34
LastEditor: JiangJi
LastEditTime: 2023-05-05 22:39:18
Discription: 
'''
import ray
import asyncio

@ray.remote(num_cpus=1)
class Collector:
    def __init__(self, cfg, data_handler=None) -> None:
        self.cfg = cfg
        self.data_handler = data_handler
    def run(self, data_server):
        while not ray.get(data_server.check_episode_limit.remote()):
            transition = ray.get(data_server.dequeue_msg.remote(msg_type="transition"))
            # print(f"transition: {transition}")
            if transition is not None:
                self.data_handler.add_transition(transition)
            training_data = self.data_handler.sample_training_data()
            # print(f"training_data: {training_data} {len(self.data_handler.buffer)}")
            if training_data is not None:
                data_server.enqueue_msg.remote(msg=training_data,msg_type="training_data")
            