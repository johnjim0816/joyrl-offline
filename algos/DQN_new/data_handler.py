#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-23 00:55:32
LastEditor: JiangJi
LastEditTime: 2023-04-24 22:24:24
Discription: 
'''
import ray
import torch
import numpy as np
@ray.remote
class DataHandler:
    def __init__(self, cfg, buffer, agent):
        self.buffer = buffer
        self.policy = agent
        
    def run(self):
        while True:
            data = self.buffer.sample()
            if data is not None:
                self.policy.update(data)
    def handle_exps_before_update(self, exps):
        exps = self.buffer.sample()
        # 将采样的数据转换为 tensor
        states = torch.tensor(np.array([exp.state for exp in exps]), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.array([exp.action for exp in exps]), device=self.device, dtype=torch.long).unsqueeze(dim=1)
        rewards = torch.tensor(np.array([exp.reward for exp in exps]), device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        next_states = torch.tensor(np.array([exp.next_state for exp in exps]), device=self.device, dtype=torch.float32)
        dones = torch.tensor(np.array([exp.done for exp in exps]), device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        data = (states, actions, rewards, next_states, dones)
        return data
