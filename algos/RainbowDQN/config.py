#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-12-24 20:41:56
LastEditor: JiangJi
LastEditTime: 2023-03-12 18:32:17
Discription:
'''
from config.config import DefaultConfig

import torch


class AlgoConfig(DefaultConfig):
    def __init__(self):
        self.gamma = 0.99  # 贴现因子，值越大，表示未来的收益占更大的比重
        self.tau = 1.0  # 软更新参数，值越小，表示在更新目标网络参数时，参数变化越小
        self.hidden_dim = 256  # 隐含层数量
        self.Vmin = 0.  # 定义值分布时使用的离散分布中的参数，表示分布所代表价值的范围下界
        self.Vmax = 200.  # 定义值分布时使用的离散分布中的参数，表示分布所代表价值的范围上界
        self.n_atoms = 51  # 定义值分布时使用的离散分布中的参数，atoms 的数量，表示离散分布的柱状数量，即价值采样点
        # self.support = torch.linspace(self.Vmin, self.Vmax, self.n_atoms)  # support of C51
        # self.delta_z = (self.Vmax - self.Vmin) / (self.n_atoms - 1)  # support of C51

        self.n_step = 1  # N-step DQN 中的 step 数
        self.batch_size = 32  # 训练 policy 及 target 模型的 batch 大小
        self.lr = 0.0001  # 学习率
        self.target_update = 200  # 同步 policy 网络和 target 网络的频率
        self.memory_capacity = 10000  # 经验回放池的大小
        self.epsilon_start = 0.95  # 探索概率的初始值
        self.epsilon_end = 0.01  # 探索概率的下界值
        self.epsilon_decay = 500  # 探索概率的衰变因子
