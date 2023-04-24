#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-17 22:40:10
LastEditor: JiangJi
LastEditTime: 2023-04-20 23:07:57
Discription: 
'''
class BaseAgent:
    def __init__(self,cfg) -> None:
        self.cfg = cfg
    def sample_action(self, state):
        pass
    def predict_action(self, state):
        pass
    def update_summary(self):
        ''' 更新 tensorboard 数据
        '''
        self.cfg.tb_writter.add_scalar(tag = f"{self.cfg.mode.lower()}_loss", scalar_value=self.loss.item(), global_step = self.update_step)
    def update(self):
        pass
    def save(self):
        pass
    def load(self):
        pass