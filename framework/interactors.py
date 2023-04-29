#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-17 13:27:23
LastEditor: JiangJi
LastEditTime: 2023-04-29 11:28:51
Discription: 
'''
import ray
import asyncio


@ray.remote(num_cpus=1)
class Interactor:
    def __init__(self, cfg, id = None, env = None, policy = None, ):
        self.cfg = cfg
        self.id = id # interactor id
        self.env = env
        self.policy = policy
    async def run(self, data_server):
        
        while not await data_server.check_episode_limit.remote():
            # print(f"Interactor {self.id} is running {await data_server.get_episode.remote()}")
            ep_reward = 0
            ep_step = 0
            await self.load_policy(data_server)
            state = self.env.reset(seed = 1)
            for _ in range(self.cfg.max_step):

                action = self.policy.get_action(state)
                next_state, reward, terminated, truncated , info = self.env.step(action)
                ep_reward += reward
                ep_step += 1
                await data_server.enqueue_exp.remote((state, action, reward, next_state, terminated, info))
                state = next_state
                if terminated:
                    print(f"Interactor {self.id} finished episode {await data_server.get_episode.remote()} with reward {ep_reward} in {ep_step} steps")
                    
                    best_reward = await data_server.get_best_reward.remote()
                    # print("info",best_reward)
                    if ep_reward > best_reward:
                        await data_server.set_best_reward.remote(ep_reward)
                        policy_params = self.policy.get_params()
                        await data_server.set_default_policy_params.remote(policy_params)
                    await data_server.increment_episode.remote()
                    break
            
        
    async def load_policy(self, data_server):
        policy_params = await data_server.dequeue_policy_params.remote()
        # print("inter policy_params",policy_params['policy_net.layers.2.0.weight'][0][:10])
        self.policy.load_params(policy_params)