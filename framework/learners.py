#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-05-07 18:30:53
LastEditor: JiangJi
LastEditTime: 2023-05-31 14:08:11
Discription: 
'''
import ray
from ray.util.queue import Queue, Empty, Full
@ray.remote
class Learner:
    ''' learner
    '''
    def __init__(self,cfg, learner_id = 0, policy=None,data_handler=None,online_tester=None) -> None:
        self.cfg = cfg
        self.learner_id = learner_id
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
    def get_model_params(self):
        ''' get model parameters
        '''
        return self.policy.get_model_params()
    def get_training_data(self,i_ep = 0):
        ''' get training data
        '''
        if self.cfg.onpolicy_flag: # on policy
            training_data = None
            if self.cfg.batch_size_flag:
                if len(self.data_handler.buffer)>=self.cfg.batch_size:
                    training_data = self.data_handler.sample_training_data()
            elif self.cfg.batch_episode_flag:
                if (i_ep+1)%self.cfg.batch_episode == 0:
                    training_data = self.data_handler.sample_training_data()
        else:
            training_data = self.data_handler.sample_training_data()
        return training_data
    def set_model_params(self,model_params):
        ''' set model parameters
        '''
        self.policy.set_model_params(model_params)
    def train(self,training_data, data_server = None, logger = None):
        ''' train policy
        '''
        if training_data is not None:
            data_server.increase_update_step.remote()
            self.update_step = ray.get(data_server.get_update_step.remote())
            self.policy.train(**training_data,update_step=self.update_step)
            self.data_handler.add_data_after_train(self.policy.data_after_train)
            if self.update_step % self.cfg.model_save_fre == 0:
                self.policy.save_model(f"{self.cfg.model_dir}/{self.update_step}")
                if self.cfg.online_eval == True:
                    best_flag, online_eval_reward = ray.get(self.online_tester.eval.remote(self.policy))
                    logger.info.remote(f"learner id: {self.learner_id}, update_step: {self.update_step}, online_eval_reward: {online_eval_reward:.3f}")
                    if best_flag:
                        logger.info.remote(f"learner {self.learner_id} for current update step obtain a better online_eval_reward: {online_eval_reward:.3f}, save the best model!")
                        self.policy.save_model(f"{self.cfg.model_dir}/best")
            return self.update_step, self.policy.summary
        return None , None

def get_ray_learner(n_gpus = 0,*args, **kwargs):
    @ray.remote(num_gpus=n_gpus)
    class RayLearner(Learner):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
    return RayLearner.remote(*args, **kwargs)