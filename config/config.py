#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-02-21 20:32:12
LastEditor: JiangJi
LastEditTime: 2023-05-15 01:10:34
Discription: 
'''
class DefaultConfig:
    def __init__(self) -> None:
        pass
    def print_cfg(self):
        print(self.__dict__)
        
class MergedConfig:
    def __init__(self) -> None:
        self.general_cfg = None
        self.algo_cfg = None
        self.env_cfg = None
class GeneralConfig():
    def __init__(self) -> None:
        self.env_name = "gym" # name of environment
        self.wrapper = None # wrapper of environment
        self.render = False # whether to render environment
        self.render_mode = "human" # 渲染模式, "human" 或者 "rgb_array"
        self.algo_name = "DQN" # name of algorithm
        self.mode = "test" # train or test
        self.mp_backend = None # multiprocessing backend: "ray", default None
        self.seed = 0 # random seed
        self.device = "cpu" # device to use
        self.max_episode = 1000 # number of episodes for training
        self.max_step = 200 # number of episodes for testing
        self.eval_eps = 10 # number of episodes for evaluation
        self.eval_per_episode = 5 # evaluation per episode
        self.max_steps = 200 # max steps for each episode
        self.load_checkpoint = True
        self.load_path = "Train_gym_BCQ_20230417-141811" # path to load model
        self.show_fig = False # show figure or not
        self.save_fig = True # save figure or not
