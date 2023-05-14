#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-07 18:30:46
LastEditor: JiangJi
LastEditTime: 2023-05-14 21:16:18
Discription: 
'''
import ray
import time
@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, cfg, id = 0 , env = None, logger = None):
        self.cfg = cfg
        self.id = id # Worker id
        self.worker_seed = self.cfg.seed + self.id
        self.env = env
        self.logger = logger
    def run(self, data_server = None, learner = None, stats_recorder = None):
        ''' Run worker
        '''
        while not ray.get(data_server.check_episode_limit.remote()): # Check if episode limit is reached
            self.ep_reward, self.ep_step = 0, 0
            self.episode = ray.get(data_server.get_episode.remote())
            state = self.env.reset(seed = self.worker_seed)
            for _ in range(self.cfg.max_step):
                action = ray.get(learner.get_action.remote(state, data_server=data_server)) # get action from learner
                next_state, reward, terminated, truncated , info = self.env.step(action) # interact with env
                self.ep_reward += reward
                self.ep_step += 1
                ray.get(learner.add_transition.remote((state, action, reward, next_state, terminated,info))) # add transition to learner
                self.update_step, self.model_summary = ray.get(learner.train.remote(data_server, stats_recorder)) # train learner
                self.add_model_summary(stats_recorder) # add model summary to stats_recorder
                state = next_state # update state
                if terminated:
                    break
            print(f"Worker {self.id} finished episode {self.episode} with reward {self.ep_reward} in {self.ep_step} steps") # debug print
            self.logger.info.remote(f"Worker {self.id} finished episode {self.episode} with reward {self.ep_reward} in {self.ep_step} steps")
            ray.get(data_server.increase_episode.remote()) # increase episode count
            self.add_interact_summary(stats_recorder) 

    def add_interact_summary(self,stats_recorder):
        ''' Add interact summary to stats_recorder
        '''
        summary = {
            'reward': self.ep_reward,
            'step': self.ep_step
        }
        ray.get(stats_recorder.add_interact_summary.remote((self.episode,summary)))

    def add_model_summary(self, stats_recorder):
        ''' Add model summary to stats_recorder
        '''
        if self.model_summary is not None:
            ray.get(stats_recorder.add_model_summary.remote((self.update_step,self.model_summary)))
