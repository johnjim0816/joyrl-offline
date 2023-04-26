#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-23 00:55:32
LastEditor: JiangJi
LastEditTime: 2023-04-26 23:19:10
Discription: 
'''
import ray
import torch
import numpy as np
from algos.base.buffers import BufferCreator
@ray.remote
class DataHandler:
    def __init__(self, cfg, agent):
        self.cfg = cfg
        self.device = cfg.device
        self.buffer = BufferCreator(cfg)()
        self.agent = agent
    def add(self, exps):
        self.buffer.push(exps)
    def update_agent(self):
        exps = self.buffer.sample(self.cfg.batch_size)
        data = self.handle_exps_before_update(exps)
        self.agent.update(**data)
    def handle_exps_before_update(self, exps):
        # 将采样的数据转换为 tensor
        states = torch.tensor(np.array([exp.state for exp in exps]), device=self.device, dtype=torch.float32)
        actions = torch.tensor(np.array([exp.action for exp in exps]), device=self.device, dtype=torch.long).unsqueeze(dim=1)
        rewards = torch.tensor(np.array([exp.reward for exp in exps]), device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        next_states = torch.tensor(np.array([exp.next_state for exp in exps]), device=self.device, dtype=torch.float32)
        dones = torch.tensor(np.array([exp.done for exp in exps]), device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones}
        return data
