#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-23 00:56:31
LastEditor: JiangJi
LastEditTime: 2023-04-26 23:20:11
Discription: 
'''
import ray
from algos.DQN.exp import Exp

@ray.remote(num_cpus=1)
class Interactor:
    def __init__(self, id, cfg, env, agent, data_handler,global_var_recorder):
        self.id = id
        self.cfg = cfg
        self.env = env
        self.data_handler = data_handler
        self.agent = agent
        self.global_var_recorder = global_var_recorder
    def create_summary(self):
        pass
    def train_one_episode(self):
        state = self.env.reset()
        ep_step = 0
        ep_reward = 0
        for _ in range(self.cfg.max_steps):
            ep_step += 1
            action = self.agent.sample_action(state)  # 采样动作
            next_state, reward, terminated, truncated , info = self.env.step(action)  # 更新环境并返回转移
            exp = [Exp(state = state, action = action, reward = reward, next_state = next_state, done = terminated, info = info)]
            self.data_handler.add.remote(exp)  # 存储样本(转移)
            state = next_state  # 更新下一个状态
            ep_reward += reward   
            if terminated:
                self.global_var_recorder.add_episode.remote(1)
                break
        i_ep = ray.get(self.global_var_recorder.read_episode.remote())
        res = {'episode': i_ep,'reward': ep_reward, 'step': ep_step}
        return res