#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-01-01 16:45:22
LastEditor: JiangJi
LastEditTime: 2023-04-19 10:45:06
Discription: 
'''
from config.config import DefaultConfig

class AlgoConfig(DefaultConfig):
    '''算法超参数类
    '''
    def __init__(self) -> None:
        ## 设置 epsilon_start=epsilon_end 可以得到固定的 epsilon，即等于epsilon_end
        self.epsilon_start = 0.95  # epsilon 初始值
        self.epsilon_end = 0.01  # epsilon 终止值
        self.epsilon_decay = 500  # epsilon 衰减率
        self.gamma = 0.95  # 奖励折扣因子
        self.lr = 0.0001  # 学习率
        self.buffer_type = "REPLAY_QUE"
        self.buffer_size = 100000  # buffer 大小
        self.batch_size = 64  # batch size
        self.target_update = 4  # target_net 更新频率
        # 神经网络层配置
        self.value_layers = [
            {'layer_type': 'embed', 'n_embeddings': [64], 'embedding_dim': [16]},
            {'layer_type': 'Linear', 'layer_dim': [64], 'activation': 'ReLU'},
            {'layer_type': 'Linear', 'layer_dim': [64], 'activation': 'ReLU'},
        ]
