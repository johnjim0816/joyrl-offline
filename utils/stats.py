#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-24 23:26:57
LastEditor: JiangJi
LastEditTime: 2023-04-24 23:28:07
Discription: 
'''
import ray

@ray.remote()
class StatsRecorder:
