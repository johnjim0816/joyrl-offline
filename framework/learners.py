#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-07 18:30:53
LastEditor: JiangJi
LastEditTime: 2023-05-15 23:38:06
Discription: 
'''
import ray
from ray.util.queue import Queue, Empty, Full
@ray.remote
class Learner:
    ''' learner
    '''
    def __init__(self,cfg,policy=None,data_handler=None,online_tester=None) -> None:
        self.cfg = cfg
        self.policy = policy
        self.data_handler = data_handler
        self.online_tester = online_tester
        self.model_params_que = Queue(maxsize=128)
    def add_transition(self,transition):
        ''' add transition to data handler
        '''
        policy_transition = self.policy.get_policy_transition()
        transition.update(policy_transition)
        self.data_handler.add_transition(transition)
    def get_action(self,state,data_server = None):
        ''' get action from policy
        '''
        ray.get(data_server.increase_sample_count.remote())
        sample_count = ray.get(data_server.get_sample_count.remote())
        return self.policy.get_action(state,sample_count=sample_count)
    
    def train(self,data_server = None, logger = None):
        ''' train policy
        '''
        training_data = self.data_handler.sample_training_data()
        if training_data is not None:
            data_server.increase_update_step.remote()
            self.update_step = ray.get(data_server.get_update_step.remote())
            self.policy.update(**training_data,update_step=self.update_step)
            if self.update_step % self.cfg.model_save_fre == 0:
                self.policy.save_model(f"{self.cfg.model_dir}/{self.update_step}")
                if self.cfg.online_eval == True:
                    best_flag, online_eval_reward = ray.get(self.online_tester.eval.remote(self.policy))
                    logger.info.remote(f"update_step: {self.update_step}, online_eval_reward: {online_eval_reward:.3f}")
                    if best_flag:
                        logger.info.remote(f"current update step obtain a better online_eval_reward: {online_eval_reward:.3f}, save the best model!")
                        self.policy.save_model(f"{self.cfg.model_dir}/best")
            return self.update_step, self.policy.summary
        return None , None


