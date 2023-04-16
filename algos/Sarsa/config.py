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
    '''算法超参数类
    '''
    def __init__(self) -> None:
        ## 设置 epsilon_start=epsilon_end 可以得到固定的 epsilon，即等于epsilon_end
        self.epsilon_start = 0.95 # epsilon 初始值
        self.epsilon_end = 0.01 # epsilon 终止值
        self.epsilon_decay = 300 # epsilon 衰减率
        self.gamma = 0.90 # 奖励折扣因子
        self.lr = 0.1 # 学习率