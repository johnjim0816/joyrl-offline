#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-17 13:27:23
LastEditor: JiangJi
LastEditTime: 2023-04-30 12:02:35
Discription: 
'''
import ray
import asyncio


@ray.remote
class Interactor:
    def __init__(self, cfg, id = None, env = None, policy = None, data_handler = None):
        self.cfg = cfg
        self.id = id # interactor id
        self.env = env
        self.policy = policy
    async def run(self, data_server, data_handler):
        
        while not await data_server.check_episode_limit.remote():
            # print(f"Interactor {self.id} is running {await data_server.get_episode.remote()}")
            ep_reward = 0
            ep_step = 0
            # await self.load_policy(data_server)
            state = self.env.reset(seed = 1)
            for _ in range(self.cfg.max_step):
                # await self.load_policy(data_server)
                action = self.policy.get_action(state)
                next_state, reward, terminated, truncated , info = self.env.step(action)
                ep_reward += reward
                ep_step += 1
                data_handler.add_transition((state, action, reward, next_state, terminated,info))
                training_data = data_handler.sample_training_data()
                if training_data:
                    self.policy.update(**training_data)
                state = next_state
                if terminated:
                    best_reward = await data_server.get_best_reward.remote()
                    print(f"Interactor {self.id} finished episode {await data_server.get_episode.remote()} with reward {ep_reward} in {ep_step} steps and best reward {best_reward}, epsilon {self.policy.epsilon}")
                    
                    if ep_reward > best_reward:
                        await data_server.set_best_reward.remote(ep_reward)
                        # policy_params = self.policy.get_params()
                        # await data_server.set_default_policy_params.remote(policy_params)
                    await data_server.increment_episode.remote()
                    break
                    

    async def load_policy(self, data_server):
        # policy_params = await data_server.get_policy_params.remote()
        policy_params = await data_server.dequeue_policy_params.remote()
        # print("inter policy_params",policy_params['policy_net.layers.2.0.weight'][0][:10])
        self.policy.load_params(policy_params)
        # for name, param in self.policy.named_parameters():
        #     param.data.copy_(policy_params[name])
            