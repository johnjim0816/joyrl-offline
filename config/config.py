#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-02-21 20:32:12
LastEditor: JiangJi
LastEditTime: 2023-05-15 13:26:18
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
        self.algo_name = "DQN" # name of algorithm
        self.mode = "train" # train, test
        self.collect_traj = False # collect trajectory or not
        # multiprocessing settings
        self.mp_backend = "single" # multiprocessing backend: "ray", default "single"
        self.n_workers = 1 # number of workers
        self.seed = 0 # random seed
        self.device = "cpu" # device to use
        self.max_episode = 100 # number of episodes for training
        self.max_step = 200 # number of episodes for testing
        # online evaluation settings
        self.online_eval = False # online evaluation or not
        self.online_eval_episode = 10 # online eval episodes
        self.load_checkpoint = True
        self.load_path = "Train_gym_BCQ_20230417-141811" # path to load model
        self.show_fig = False # show figure or not
        self.save_fig = True # save figure or not
