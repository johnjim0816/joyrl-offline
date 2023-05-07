#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-07 18:30:46
LastEditor: JiangJi
LastEditTime: 2023-05-07 19:50:31
Discription: 
'''
import ray
import time
@ray.remote(num_cpus=1)
class Trainer:
    def __init__(self, cfg, id = None, env = None, policy = None):
        self.cfg = cfg
        self.id = id # interactor id
        self.env = env
        self.policy = policy
    def run(self, data_server = None, agent = None):
        while not ray.get(data_server.check_episode_limit.remote()):
            # print(f"Interactor {self.id} is running {await data_server.get_episode.remote()}")
            ep_reward = 0
            ep_step = 0
            state = self.env.reset(seed = 1)
            print(f"Worker {self.id} start episode {ray.get(data_server.get_episode.remote())}")
            for _ in range(self.cfg.max_step):
                data_server.increase_sample_count.remote()
                sample_count = ray.get(data_server.get_sample_count.remote())
                action = ray.get(agent.get_action.remote(state, sample_count=sample_count))
                # action = self.policy.get_action(state, sample_count=sample_count)
                next_state, reward, terminated, truncated , info = self.env.step(action)
                ep_reward += reward
                ep_step += 1
                transition = (state, action, reward, next_state, terminated,info)
                agent.add_transition.remote(transition)
                agent.train.remote()
                # time.sleep(0.1)
                state = next_state
                if terminated:
                    print(f"Worker {self.id} finished episode {ray.get(data_server.get_episode.remote())} with reward {ep_reward}")
                    data_server.increment_episode.remote()
                    break
                