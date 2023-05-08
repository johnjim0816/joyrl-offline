#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-07 18:30:53
LastEditor: JiangJi
LastEditTime: 2023-05-08 23:51:40
Discription: 
'''
import ray
@ray.remote()
class Learner:
    def __init__(self,cfg,policy=None,data_handler=None) -> None:
        self.cfg = cfg
        self.policy = policy
        self.data_handler = data_handler
    def add_transition(self,transition):
        self.data_handler.add_transition(transition)
    def get_training_data(self):
        training_data = self.data_handler.sample_training_data()
        return training_data
    def get_action(self,state,data_server = None):
        data_server.increase_sample_count.remote()
        sample_count = ray.get(data_server.get_sample_count.remote())
        return self.policy.get_action(state,sample_count=sample_count)
    def get_policy(self):
        policy_params = self.policy.get_policy_params()
        optimizer_params = self.policy.get_optimizer_params()
        return policy_params,optimizer_params
    def set_policy(self,policy_params,optimizer_params):
        self.policy.load_policy_params(policy_params)
        self.policy.load_optimizer_params(optimizer_params)
   
    def train(self,data_server = None, stats_recorder = None):
        # print("Learner is training")
       
        training_data = self.data_handler.sample_training_data()
        if training_data is not None:
            data_server.increase_update_step.remote()
            self.update_step = ray.get(data_server.get_update_step.remote())
            self.policy.update(**training_data,update_step=self.update_step)
            self.add_policy_summary(stats_recorder)
            
    def add_policy_summary(self, stats_recorder):
        summary = self.policy.summary
        stats_recorder.add_policy_summary.remote((self.update_step,summary))

