#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 16:16:04
LastEditor: JiangJi
LastEditTime: 2023-05-15 21:42:21
Discription: 
'''
import ray
from ray.util.queue import Queue, Empty, Full

class BaseDataServer:
    def __init__(self,cfg) -> None:
        self.global_episode = 0 # current global episode
        self.global_sample_count = 0 # global sample count
        self.global_update_step = 0 # global update step
        self.max_episode = cfg.max_episode # max episode
    def increase_episode(self, i=1):
        ''' increase episode
        '''
        self.global_episode += i
    def get_episode(self):
        ''' get current episode
        '''
        return self.global_episode
    def check_task_end(self):
        ''' check if episode reaches the max episode
        '''
        return self.global_episode >= self.max_episode >=0
    def increase_sample_count(self, i = 1):
        ''' increase sample count
        '''
        self.global_sample_count += i
    def get_sample_count(self):
        ''' get sample count
        '''
        return self.global_sample_count
    def increase_update_step(self, i = 1):
        ''' increase update step
        '''
        self.global_update_step += i
    def get_update_step(self):
        ''' get update step
        '''
        return self.global_update_step
    
class SimpleDataServer(BaseDataServer):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)
        self.ep_frames = [] # episode frames for visualization
    def add_ep_frame(self, ep_frame):
        ''' add one step frame
        '''
        self.ep_frames.append(ep_frame)
    def get_ep_frames(self):
        ''' get episode frames
        '''
        return self.ep_frames
@ray.remote
class DataServer:
    def __init__(self, cfg) -> None:
        self.curr_episode = 0 # current episode
        self.sample_count = 0 # sample count
        self.update_step = 0 # update step
        self.max_episode = cfg.max_episode
    def increase_episode(self):
        ''' increase episode
        '''
        self.curr_episode += 1
    def check_episode_limit(self):
        ''' check if episode reaches the max episode
        '''
        return self.curr_episode > self.max_episode
    def get_episode(self):
        ''' get current episode
        '''
        return self.curr_episode
    def increase_sample_count(self, i = 1):
        ''' increase sample count
        '''
        self.sample_count += i
    def get_sample_count(self):
        ''' get sample count
        '''
        return self.sample_count
    def increase_update_step(self):
        ''' increase update step
        '''
        self.update_step += 1
    def get_update_step(self):
        ''' get update step
        '''
        return self.update_step

