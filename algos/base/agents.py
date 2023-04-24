#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-04-17 22:40:10
LastEditor: JiangJi
LastEditTime: 2023-04-17 22:40:10
Discription: 
'''
class BaseAgent:
    def __init__(self) -> None:
        pass
    def sample_action(self, state):
        pass
    def predict_action(self, state):
        pass
    def update(self):
        pass
    def save(self):
        pass
    def load(self):
        pass