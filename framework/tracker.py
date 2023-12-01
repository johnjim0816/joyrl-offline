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
from framework.message import Msg, MsgType
class BaseTracker:
    def __init__(self,cfg) -> None:
        self.global_episode = 0 # current global episode
        self.global_sample_count = 0 # global sample count
        self.global_update_step = 0 # global update step
        self.max_episode = cfg.max_episode # max episode

    def pub_msg(self, msg: Msg):
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.DATASERVER_GET_EPISODE:
            return self._get_episode()
        elif msg_type == MsgType.DATASERVER_INCREASE_EPISODE:
            episode_delta = 1 if msg_data is None else msg_data
            self._increase_episode(i = episode_delta)
        # elif msg_type == MsgType.GET_SAMPLE_COUNT:
        #     self._get_sample_count(msg_data)
        elif msg_type == MsgType.DATASERVER_GET_UPDATE_STEP:
            return self._get_update_step()
        # elif msg_type == MsgType.CHECK_TASK_END:
        #     self._check_task_end(msg_data)
        # elif msg_type == MsgType.INCREASE_SAMPLE_COUNT:
        #     self._increase_sample_count(msg_data)
        elif msg_type == MsgType.DATASERVER_INCREASE_UPDATE_STEP:
            update_step_delta = 1 if msg_data is None else msg_data
            self._increase_update_step(i = update_step_delta)
            
        elif msg_type == MsgType.DATASERVER_CHECK_TASK_END:
            return self._check_task_end()
        else:
            raise NotImplementedError

    def _increase_episode(self, i: int =1):
        ''' increase episode
        '''
        self.global_episode += i
    def _get_episode(self):
        ''' get current episode
        '''
        return self.global_episode
    
    def _check_task_end(self):
        ''' check if episode reaches the max episode
        '''
        if self.max_episode < 0:
            return False
        return self.global_episode >= self.max_episode 
    
    def increase_sample_count(self, i = 1):
        ''' increase sample count
        '''
        self.global_sample_count += i

    def get_sample_count(self):
        ''' get sample count
        '''
        return self.global_sample_count
    
    def _increase_update_step(self, i: int =1):
        ''' increase update step
        '''
        self.global_update_step += i
        
    def _get_update_step(self):
        ''' get update step
        '''
        return self.global_update_step
    
class SimpleTracker(BaseTracker):
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
class RayTracker(BaseTracker):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)

    