#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-07 18:30:46
LastEditor: JiangJi
LastEditTime: 2023-05-15 23:39:17
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
            state, info = self.env.reset(seed = self.worker_seed)
            for _ in range(self.cfg.max_step):
                action = ray.get(learner.get_action.remote(state, data_server=data_server)) # get action from learner
                next_state, reward, terminated, truncated , info = self.env.step(action) # interact with env
                self.ep_reward += reward
                self.ep_step += 1
                ray.get(learner.add_transition.remote((state, action, reward, next_state, terminated,info))) # add transition to learner
                self.update_step, self.model_summary = ray.get(learner.train.remote(data_server, logger = self.logger)) # train learner
                self.add_model_summary(stats_recorder) # add model summary to stats_recorder
                state = next_state # update state
                if terminated:
                    break
            self.logger.info.remote(f"Worker {self.id} finished episode {self.episode} with reward {self.ep_reward} in {self.ep_step} steps")
            ray.get(data_server.increase_episode.remote()) # increase episode count
            self.add_interact_summary(stats_recorder)  # add interact summary to stats_recorder

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

class SimpleTester:
    ''' Simple online tester
    '''
    def __init__(self,cfg,env=None) -> None:
        self.cfg = cfg
        self.env = env
        self.best_eval_reward = -float('inf')
    def eval(self,policy):
        ''' Evaluate policy
        '''
        sum_eval_reward = 0
        for _ in range(self.cfg.online_eval_episode):
            state, info = self.env.reset(seed = self.cfg.seed)
            ep_reward, ep_step = 0, 0 # reward per episode, step per episode
            while True:
                action = policy.predict_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = next_state
                ep_reward += reward
                ep_step += 1
                if terminated or (0<= self.cfg.max_step <= ep_step):
                    break
            sum_eval_reward += ep_reward
        mean_eval_reward = sum_eval_reward / self.cfg.online_eval_episode
        if mean_eval_reward > self.best_eval_reward:
            self.best_eval_reward = mean_eval_reward
            return True, mean_eval_reward
        return False, mean_eval_reward
@ray.remote    
class RayTester(SimpleTester):
    ''' Ray online tester
    '''
    def __init__(self,cfg,env=None) -> None:
        super().__init__(cfg,env)
    