#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 16:18:44
LastEditor: JiangJi
LastEditTime: 2023-04-28 21:34:25
Discription: 
'''
import ray 
@ray.remote
class StatsRecorder:
    def __init__(self, cfg) -> None:
        self.id = 0
    async def run(self, data_server):
        self.id = 1
