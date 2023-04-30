#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-27 23:45:49
LastEditor: JiangJi
LastEditTime: 2023-04-30 11:52:19
Discription: 
'''
import ray 
import asyncio
import gym
import time
@ray.remote(num_cpus=1)
class Learner:
    def __init__(self, cfg, id = None, policy = None ) -> None:
        self.cfg = cfg
        self.id = id
        self.policy = policy
        self.env = gym.make("CartPole-v1", new_step_api=True)
    async def run(self, data_server, data_handler):
        while not await data_server.check_episode_limit.remote():
            training_data = data_handler.sample_training_data()
            if training_data:
                self.policy.update(**training_data)
                policy_params = self.policy.get_params()
                await data_server.enqueue_policy_params.remote(policy_params)
            # await data_server.set_policy_params.remote(self.policy.get_params())
            # print(f"learner update time {time.time()-s_t}")
            # self.policy.update(**training_data)
            # self.test()
            # policy_params = self.policy.get_params()
            # # print("learner policy_params",policy_params['policy_net.layers.2.0.weight'][0][:10])
            # await data_server.enqueue_policy_params.remote(policy_params)
            # await data_server.set_default_policy_params.remote(policy_params)
            
    def test(self):
        state = self.env.reset()
        ep_reward = 0
        for  _ in range(self.cfg.max_step):
            action = self.policy.get_action(state)
            next_state, reward, terminated, truncated , info = self.env.step(action)
            state = next_state
            ep_reward += reward
            if terminated:
                print(f"learner test reward {ep_reward}")
                break 
