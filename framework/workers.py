#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-07 18:30:46
LastEditor: JiangJi
LastEditTime: 2023-05-07 21:47:28
Discription: 
'''
import ray

@ray.remote(num_cpus=1)
class Worker:
    def __init__(self, cfg, id = None, env = None, policy = None):
        self.cfg = cfg
        self.id = id # interactor id
        self.env = env
        self.policy = policy
    def run(self, data_server = None, learner = None):
        while not ray.get(data_server.check_episode_limit.remote()):
            # print(f"Interactor {self.id} is running {await data_server.get_episode.remote()}")
            ep_reward = 0
            ep_step = 0
            state = self.env.reset(seed = 1)
            for _ in range(self.cfg.max_step):
                action = ray.get(learner.get_action.remote(state, data_server = data_server))
                next_state, reward, terminated, truncated , info = self.env.step(action)
                ep_reward += reward
                ep_step += 1
                learner.add_transition.remote((state, action, reward, next_state, terminated,info))
                learner.train.remote(data_server)
                state = next_state
                if terminated:
                    break
            print(f"Worker {self.id} finished episode {ray.get(data_server.get_episode.remote())} with reward {ep_reward}")
            data_server.increase_episode.remote()