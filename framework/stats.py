#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-28 16:18:44
LastEditor: JiangJi
LastEditTime: 2023-05-14 22:32:36
Discription: 
'''
import ray 
from ray.util.queue import Queue, Empty, Full
from pathlib import Path
import logging
from torch.utils.tensorboard import SummaryWriter  
@ray.remote
class StatsRecorder:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.interact_summary_que = Queue(maxsize=128)
        self.model_summary_que = Queue(maxsize=128)
        self.interact_writter = SummaryWriter(log_dir=f"{self.cfg.tb_dir}/interact")
        self.policy_writter = SummaryWriter(log_dir=f"{self.cfg.tb_dir}/model")
    def add_interact_summary(self,summary):
        self.interact_summary_que.put(summary, block=False)
        self.write_interact_summary()
    def add_model_summary(self,summary):
        self.model_summary_que.put(summary, block=False) 
        self.write_model_summary()
    def write_interact_summary(self):
        while self.interact_summary_que.qsize() > 0:
            episode,interact_summary = self.interact_summary_que.get()
            for key, value in interact_summary.items():
                self.interact_writter.add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value=value, global_step = episode)
    def write_model_summary(self):
        while self.model_summary_que.qsize() > 0:
            update_step, model_summary = self.model_summary_que.get()
            for key, value in model_summary['scalar'].items():
                self.policy_writter.add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value=value, global_step = update_step)
class BaseLogger(object):
    def __init__(self, fpath = None) -> None:
        Path(fpath).mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(name="BaseLog")  
        self.logger.setLevel(logging.INFO)
        self.formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s: - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')
        # output to file by using FileHandler
        fh = logging.FileHandler(f"{fpath}/log.txt")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)
    def info(self, msg):
        self.logger.info(msg)

class SimpleLogger(BaseLogger):
    def __init__(self, fpath = None) -> None:
        super().__init__(fpath)
        self.logger.name = "SimpleLog"
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

@ray.remote(num_cpus=0)
class RayLogger(BaseLogger):
    def __init__(self, fpath=None) -> None:
        super().__init__(fpath)
        self.logger.name = "RayLog"
