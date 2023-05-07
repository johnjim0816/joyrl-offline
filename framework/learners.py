#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-27 23:45:49
LastEditor: JiangJi
LastEditTime: 2023-05-07 15:35:18
Discription: 
'''
import ray 
import time
@ray.remote(num_cpus=1)
class Learner:
    def __init__(self, cfg, id = None, policy = None ) -> None:
        self.cfg = cfg
        self.id = id
        self.policy = policy
        self.i = 0
    def run(self, data_server):
        while not ray.get(data_server.check_episode_limit.remote()):
            self.i += 1
            training_data = ray.get(data_server.dequeue_msg.remote(msg_type="training_data"))
            if training_data:
                self.policy.update(**training_data)
                policy_params = self.policy.get_params()
                # ray.get(data_server.enqueue_msg.remote(msg = policy_params, msg_type="policy_params"))
                while not ray.get(data_server.enqueue_msg.remote(msg = policy_params, msg_type="policy_params")):
                    time.sleep(0.05)
                
            # print(f"learner is running {self.i}, {training_data==None}")