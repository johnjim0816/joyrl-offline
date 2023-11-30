import ray
import torch
import time
import copy
import os
import threading
class BaseTester:
    ''' Base class for online tester
    '''
    def __init__(self, cfg, env = None, policy = None, *args, **kwargs) -> None:
        self.cfg = cfg
        self.env = copy.deepcopy(env)
        self.policy = copy.deepcopy(policy)
        self.logger = kwargs['logger']
        self.best_eval_reward = -float('inf')

    def run(self, policy, *args, **kwargs):
        ''' Run online tester
        '''
        ''' Evaluate policy
        '''
        raise NotImplementedError
class SimpleTester(BaseTester):
    ''' Simple online tester
    '''
    def __init__(self, cfg, env = None, policy = None, *args, **kwargs) -> None:
        super().__init__(cfg, env, policy, *args, **kwargs)
        self.curr_test_step = -1
        self._thread_eval_policy = threading.Thread(target=self._eval_policy)
        self._thread_eval_policy.setDaemon(True)
        self.start()

    def _check_updated_model(self):

        model_step_list = os.listdir(self.cfg.model_dir)
        model_step_list = [int(model_step) for model_step in model_step_list if model_step.isdigit()]
        model_step_list.sort()
        if len(model_step_list) == 0:
            return False, -1
        elif model_step_list[-1] == self.curr_test_step:
            return False, -1
        elif model_step_list[-1] > self.curr_test_step:
            return True, model_step_list[-1]

    def start(self):
        self._thread_eval_policy.start()

    def _eval_policy(self):
        ''' Evaluate policy
        '''
        while True:
            updated, model_step = self._check_updated_model()
            if updated:
                self.curr_test_step = model_step
                model_params = torch.load(f"{self.cfg.model_dir}/{self.curr_test_step}")
                self.policy.put_model_params(model_params)
                sum_eval_reward = 0
                for _ in range(self.cfg.online_eval_episode):
                    state, info = self.env.reset()
                    ep_reward, ep_step = 0, 0
                    while True:
                        action = self.policy.get_action(state, mode = 'predict')
                        next_state, reward, terminated, truncated, info = self.env.step(action)
                        state = next_state
                        ep_reward += reward
                        ep_step += 1
                        if terminated or truncated or (0<= self.cfg.max_step <= ep_step):
                            sum_eval_reward += ep_reward
                            break
                mean_eval_reward = sum_eval_reward / self.cfg.online_eval_episode
                self.logger.info(f"test_step: {self.curr_test_step}, online_eval_reward: {mean_eval_reward:.3f}")
                if mean_eval_reward >= self.best_eval_reward:
                    self.logger.info(f"current test step obtain a better online_eval_reward: {mean_eval_reward:.3f}, save the best model!")
                    torch.save(model_params, f"{self.cfg.model_dir}/best")
                    self.best_eval_reward = mean_eval_reward
            time.sleep(1)
    
@ray.remote
class RayTester(BaseTester):
    ''' Ray online tester
    '''
    def __init__(self, cfg, env=None) -> None:
        super().__init__(cfg,env)

    def eval(self, policy, global_update_step = 0, logger = None):
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
        logger.info.remote(f"update_step: {global_update_step}, online_eval_reward: {mean_eval_reward:.3f}")
        if mean_eval_reward >= self.best_eval_reward:
            logger.info.remote(f"current update step obtain a better online_eval_reward: {mean_eval_reward:.3f}, save the best model!")
            policy.save_model(f"{self.cfg.model_dir}/best")
            self.best_eval_reward = mean_eval_reward
        summary_data = [(global_update_step,{"online_eval_reward": mean_eval_reward})]
        output = {"summary":summary_data}
        return output

    def run(self, policy, *args, **kwargs):
        ''' Run online tester
        '''
        dataserver, logger = kwargs['dataserver'], kwargs['logger']
        global_update_step = ray.get(dataserver.get_update_step.remote()) # get global update step
        if global_update_step % self.cfg.model_save_fre == 0 and self.cfg.online_eval == True:
            return self.eval(policy, global_update_step = global_update_step, logger = logger)