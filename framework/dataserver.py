#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 16:16:04
LastEditor: JiangJi
LastEditTime: 2023-05-07 15:04:16
Discription: 
'''
import ray
from ray.util.queue import Queue, Empty, Full

@ray.remote
class DataServer:
    def __init__(self, cfg) -> None:
        self.curr_episode = 0
        self.max_epsiode = cfg.max_epsiode
        self.exps_que = Queue()
        self.policy_params_que = Queue()
        self.training_data_que = Queue()
        self.best_reward = -float('inf')
        self.policy_params = None
    def increment_episode(self):
        self.curr_episode += 1
    def check_episode_limit(self):
        return self.curr_episode >= self.max_epsiode
    def get_episode(self):
        return self.curr_episode
    def enqueue_msg(self, msg, msg_type = None):
        try:
            if msg_type == "transition":
                self.exps_que.put(msg, block=False)
            elif msg_type == "training_data":
                self.training_data_que.put(msg, block=False)
            elif msg_type == "policy_params":
                self.policy_params_que.put(msg, block=False)
            else: 
                raise NotImplementedError
            return True
        except Full:
            return False 
    def dequeue_msg(self, msg_type = None):
        # print(len(self.exps_que),len(self.training_data_que),len(self.policy_params_que))
        try:
            if msg_type == "transition":
                return self.exps_que.get() if self.exps_que.qsize() > 0 else None
            elif msg_type == "training_data":
                return self.training_data_que.get() if self.training_data_que.qsize() > 0 else None
            elif msg_type == "policy_params":
                return self.policy_params_que.get() if self.policy_params_que.qsize() > 0 else None
            else:
                raise NotImplementedError
        except Empty:
            print(f"{msg_type} empty")
            
            return None
    def set_policy_params(self, policy_params):
        self.policy_params = policy_params
    def get_policy_params(self):
        return self.policy_params
    def set_best_reward(self, reward):
        self.best_reward = reward
    def get_best_reward(self):
        return self.best_reward
