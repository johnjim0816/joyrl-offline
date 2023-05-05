#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-17 13:27:23
LastEditor: JiangJi
LastEditTime: 2023-05-05 23:22:53
Discription: 
'''
import ray


@ray.remote
class Interactor:
    def __init__(self, cfg, id = None, env = None, policy = None, data_handler = None):
        self.cfg = cfg
        self.id = id # interactor id
        self.env = env
        self.policy = policy
    def run(self, data_server,):
        while not ray.get(data_server.check_episode_limit.remote()):
            # print(f"Interactor {self.id} is running {await data_server.get_episode.remote()}")
            ep_reward = 0
            ep_step = 0
            import time
            state = self.env.reset(seed = 1)

            for _ in range(self.cfg.max_step):
                s_t = time.time()
                action = self.policy.get_action(state)
                next_state, reward, terminated, truncated , info = self.env.step(action)
                ep_reward += reward
                ep_step += 1
                transition = (state, action, reward, next_state, terminated,info)
                data_server.enqueue_msg.remote(msg = transition, msg_type = "transition")
                self.load_policy(data_server)
                state = next_state
                if terminated:
                    print(f"Interactor {self.id} finished episode {ray.get(data_server.get_episode.remote())} with reward {ep_reward} in {ep_step} steps, epsilon {self.policy.epsilon}")
                    data_server.increment_episode.remote()
                    break
                
    def load_policy(self, data_server):
        policy_params = ray.get(data_server.dequeue_msg.remote(msg_type="policy_params"))
        if policy_params is not None:
            self.policy.load_params(policy_params)
