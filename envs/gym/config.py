#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-13 00:10:26
LastEditor: JiangJi
LastEditTime: 2023-04-13 00:54:11
Discription: 
'''
from config.config import DefaultConfig

class EnvConfig(DefaultConfig):
    def __init__(self) -> None:
        super().__init__()
        self.id = "HalfCheetah-v3" # 环境名称
        self.new_step_api = True # 是否使用新的step api
        self.render_mode = None # 渲染模式, None, rgb_array, human