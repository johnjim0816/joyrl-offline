#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-03-10 23:35:37
LastEditor: JiangJi
LastEditTime: 2023-03-10 23:37:58
Discription: 
'''
class AlgoConfig:
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.95 # epsilon start value
        self.epsilon_end = 0.01 # epsilon end value
        self.epsilon_decay = 300 # epsilon decay rate
        self.gamma = 0.90 # discount factor
        self.lr = 0.1 # learning rate