#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 16:18:44
LastEditor: JiangJi
LastEditTime: 2023-05-08 00:14:25
Discription: 
'''
import ray 
from ray.util.queue import Queue, Empty, Full
from torch.utils.tensorboard import SummaryWriter  
@ray.remote
class StatsRecorder:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.interact_summary_que = Queue(maxsize=128)
        self.policy_summary_que = Queue(maxsize=128)
        self.interact_writter = SummaryWriter(log_dir=f"{self.cfg.tb_dir}/interact")
        self.policy_writter = SummaryWriter(log_dir=f"{self.cfg.tb_dir}/policy")
    def add_interact_summary(self,summary):
        self.interact_summary_que.put(summary, block=False)
        self.write_interact_summary()
    def add_policy_summary(self,summary):
        self.policy_summary_que.put(summary, block=False) 
        self.write_policy_summary()
    def write_interact_summary(self):
        while self.interact_summary_que.qsize() > 0:
            episode,interact_summary = self.interact_summary_que.get()
            for key, value in interact_summary.items():
                self.interact_writter.add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value=value, global_step = episode)
    def write_policy_summary(self):
        while self.policy_summary_que.qsize() > 0:
            update_step, policy_summary = self.policy_summary_que.get()
            for key, value in policy_summary['scalar'].items():
                self.policy_writter.add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value=value, global_step = update_step)
    

