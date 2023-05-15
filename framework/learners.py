#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-07 18:30:53
LastEditor: JiangJi
LastEditTime: 2023-05-14 20:18:45
Discription: 
'''
import ray
from ray.util.queue import Queue, Empty, Full
@ray.remote
class Learner:
    def __init__(self,cfg,policy=None,data_handler=None) -> None:
        self.cfg = cfg
        self.policy = policy
        self.data_handler = data_handler
        self.model_params_que = Queue(maxsize=128)
    def add_transition(self,transition):
        self.data_handler.add_transition(transition)
    def get_action(self,state,data_server = None):
        ray.get(data_server.increase_sample_count.remote())
        sample_count = ray.get(data_server.get_sample_count.remote())
        return self.policy.get_action(state,sample_count=sample_count)
    def train(self,data_server = None, stats_recorder = None):
        training_data = self.data_handler.sample_training_data()
        if training_data is not None:
            data_server.increase_update_step.remote()
            self.update_step = ray.get(data_server.get_update_step.remote())
            self.policy.update(**training_data,update_step=self.update_step)
            return self.update_step, self.policy.summary
        return None , None


