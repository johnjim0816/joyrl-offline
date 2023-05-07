#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-07 18:30:53
LastEditor: JiangJi
LastEditTime: 2023-05-07 19:43:34
Discription: 
'''
import ray
import time
@ray.remote
class Agent:
    def __init__(self,cfg,policy=None,data_handler=None) -> None:
        self.cfg = cfg
        self.policy = policy
        self.data_handler = data_handler
    def add_transition(self,transition):
        self.data_handler.add_transition(transition)
    def get_action(self,state,sample_count = None):
        return self.policy.get_action(state,sample_count=sample_count)
    def train(self):
        # print("Agent is training")
        training_data = self.data_handler.sample_training_data()
        if training_data is not None:
            self.policy.update(**training_data)