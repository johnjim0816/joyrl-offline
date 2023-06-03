
import ray
import numpy as np
from framework.interactors import Interactor

@ray.remote
class Worker:
    def __init__(self, cfg, id = 0 , env = None, logger = None):
        self.cfg = cfg
        self.id = id
        self.interactor = Interactor(cfg, env, id = self.id) # Interactor
        self.logger = logger
    def run(self, data_server = None, learners = None, stats_recorder = None):
        ''' Run worker
        '''
        while not ray.get(data_server.check_episode_limit.remote()): # Check if episode limit is reached
            policy = ray.get(learners[self.learner_id].get_policy.remote()) # get policy from learner
            run_sample_count = ray.get(data_server.get_sample_count.remote()) # get sample count from data server
            if self.cfg.onpolicy_flag: # on policy
                if self.cfg.batch_size_flag: output = self.interactor.run(policy,sample_count = run_sample_count, n_steps=self.cfg.batch_size)
                if self.cfg.batch_episode_flag: output = self.interactor.run(policy,sample_count = run_sample_count, n_episodes=self.cfg.batch_episode)
            else: # off policy
                output = self.interactor.run(policy,sample_count = run_sample_count, n_steps = 1)
            ray.get(data_server.increase_sample_count.remote()) # increase sample count
            self.add_interact_summary(output['interact_summary'], data_server, stats_recorder)
            if self.cfg.share_buffer: # if all learners share the same buffer
                ray.get(learners[0].add_exps.remote(output['exps'])) # add transition to learner
                training_data = ray.get(learners[0].get_training_data.remote()) # get training data from learner
            else:
                ray.get(learners[self.learner_id].add_exps.remote(output['exps'])) # add transition to data server
                training_data = ray.get(learners[self.learner_id].get_training_data.remote()) # get training data from data 
            self.update_step, self.model_summary = ray.get(learners[self.learner_id].learn.remote(training_data, data_server=data_server, logger = self.logger)) # train learner
            self.broadcast_model_params(learners) # broadcast model parameters to data server
            self.add_model_summary(stats_recorder) # add model summary to stats_recorder
    def broadcast_model_params(self, learners = None):
        ''' Broadcast model parameters to data server
        '''
        #  aggregation model parameters
        # import torch
        # all_model_params = []
        # for learner in learners:
        #     all_model_params.append(ray.get(learner.get_model_params.remote()))
        # average_model_params = {}
        # for key in all_model_params[0].keys():
        #     average_model_params[key] = torch.mean(torch.stack([state_dict[key] for state_dict in all_model_params]), dim=0)
        # for learner in learners:
        #     ray.get(learner.set_model_params.remote(average_model_params))
        # broadcast model parameters
        # if self.learner_id == 0:
        if self.cfg.n_learners > 1:
            model_params = ray.get(learners[0].get_model_params.remote()) # 0 is the main learner
            for learner in learners[1:]:
                ray.get(learner.set_model_params.remote(model_params))
    def set_learner_id(self,learner_id):
        ''' Set learner id
        '''
        self.learner_id = learner_id
    
    def add_interact_summary(self, interact_summary, data_server, stats_recorder):
        ''' Add interact summary to stats_recorder
        '''
        if len(interact_summary['reward']) == 0: return 
        curr_episode = ray.get(data_server.get_episode.remote()) # get current episode
        for i in range(len(interact_summary['reward'])):
            curr_episode += 1
            reward, step = interact_summary['reward'][i], interact_summary['step'][i]
            summary = {'reward': reward, 'step': step}
            ray.get(stats_recorder.add_interact_summary.remote((curr_episode,summary)))
            ray.get(data_server.increase_episode.remote())
            self.logger.info.remote(f"Worker {self.id} finished episode {curr_episode} with reward {reward:.3f} in {step} steps")

    def add_model_summary(self, stats_recorder):
        ''' Add model summary to stats_recorder
        '''
        if self.model_summary is not None:
            ray.get(stats_recorder.add_model_summary.remote((self.update_step,self.model_summary)))

class SimpleTester:
    ''' Simple online tester
    '''
    def __init__(self,cfg,env=None) -> None:
        self.cfg = cfg
        self.env = env
        self.best_eval_reward = -float('inf')
    def eval(self,policy):
        ''' Evaluate policy
        '''
        sum_eval_reward = 0
        for _ in range(self.cfg.online_eval_episode):
            state, info = self.env.reset(seed = self.cfg.seed)
            ep_reward, ep_step = 0, 0 # reward per episode, step per episode
            while True:
                action = policy.get_action(state, mode = 'predict')
                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = next_state
                ep_reward += reward
                ep_step += 1
                if terminated or (0<= self.cfg.max_step <= ep_step):
                    sum_eval_reward += ep_reward
                    break
        mean_eval_reward = sum_eval_reward / self.cfg.online_eval_episode
        if mean_eval_reward >= self.best_eval_reward:
            self.best_eval_reward = mean_eval_reward
            return True, mean_eval_reward
        return False, mean_eval_reward
@ray.remote
class RayTester(SimpleTester):
    ''' Ray online tester
    '''
    def __init__(self,cfg,env=None) -> None:
        super().__init__(cfg,env)
    
def get_ray_tester(n_gpus = 0, *args, **kwargs):
    ''' Get ray online tester
    '''
    if n_gpus > 0:
        return RayTester.options(num_gpus=n_gpus).remote(*args, **kwargs)
    else:
        return RayTester.remote(*args, **kwargs)