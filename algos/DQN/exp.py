#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-16 23:56:37
LastEditor: JiangJi
LastEditTime: 2023-04-16 23:57:02
Discription: 
'''
from algos.base.exps import BaseExp

class Exp(BaseExp):
    def __init__(self, state=None, action=None, reward=None, next_state=None, done=None, **kwargs) -> None:
        super().__init__(state, action, reward, next_state, done, **kwargs)
        self.info = kwargs.get('info', None)