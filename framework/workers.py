#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-07 18:30:46
LastEditor: JiangJi
LastEditTime: 2023-05-08 00:18:26
Discription: 
'''
import ray

@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, cfg, id = None, env = None, policy = None):
        self.cfg = cfg
        self.id = id # interactor id
        self.worker_seed = self.cfg.seed + self.id
        self.env = env
        self.policy = policy
    def run(self, data_server = None, learner = None, stats_recorder = None):
        while not ray.get(data_server.check_episode_limit.remote()):
            # print(f"Interactor {self.id} is running {await data_server.get_episode.remote()}")
            self.ep_reward, self.ep_step = 0, 0
            self.episode = ray.get(data_server.get_episode.remote())
            state = self.env.reset(seed = self.worker_seed)
            for _ in range(self.cfg.max_step):
                action = ray.get(learner.get_action.remote(state, data_server = data_server))
                next_state, reward, terminated, truncated , info = self.env.step(action)
                self.ep_reward += reward
                self.ep_step += 1
                learner.add_transition.remote((state, action, reward, next_state, terminated,info))
                learner.train.remote(data_server, stats_recorder)
                state = next_state
                if terminated:
                    break
            
            print(f"Worker {self.id} finished episode {self.episode} with reward {self.ep_reward} in {self.ep_step} steps")
            data_server.increase_episode.remote()
            self.add_interact_summary(stats_recorder)
    def add_interact_summary(self,stats_recorder):
        summary = {
            'reward': self.ep_reward,
            'step': self.ep_step
        }
        stats_recorder.add_interact_summary.remote((self.episode,summary))