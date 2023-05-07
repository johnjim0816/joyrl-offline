#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 16:18:44
LastEditor: JiangJi
LastEditTime: 2023-05-07 23:21:07
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
    def add_summary(self,summary, summary_type = None):
        if summary_type == "interact":
            self.interact_summary_que.put(summary, block=False)
        elif summary_type == "policy":
            self.policy_summary_que.put(summary, block=False)
        else: 
            raise NotImplementedError
        if self.interact_summary_que.qsize() > 0 or self.policy_summary_que.qsize() > 0:
            self.write_summary()
    def run(self,data_server_handle):
        while not data_server_handle.check_episode_limit.remote():
            self.write_summary()
    def write_summary(self):
        if self.interact_summary_que.qsize() > 0:
            episode,interact_summary = self.interact_summary_que.get()
            for key, value in interact_summary.items():
                self.interact_writter.add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value=value, global_step = episode)
        if self.policy_summary_que.qsize() > 0:
            update_step, policy_summary = self.policy_summary_que.get()
            for key, value in policy_summary['scalar'].items():
                self.policy_writter.add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value=value, global_step = update_step)
    

