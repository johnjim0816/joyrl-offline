import gymnasium as gym
from typing import Tuple

from algos.base.exps import Exp
from envs.base.config import BaseEnvConfig

class BaseInteractor:
    ''' Interactor for gym env to support sample n-steps or n-episodes traning data
    '''
    def __init__(self, cfg: BaseEnvConfig, id = 0, *args, **kwargs) -> None:
        self.cfg = cfg 
        self.id = id
        self.env = gym.make(self.cfg.env_cfg.id)
        self.seed = self.cfg.seed + self.id
        self.data = None
        self.reset_summary()
        self.reset_ep_params()
        self.init()

    def reset_summary(self):
        ''' Create interact summary
        '''
        self.summary = list()  

    def update_summary(self, summary: Tuple):
        ''' Add interact summary
        '''
        self.summary.append(summary)

    def get_summary(self):
        return self.summary
    
    def reset_ep_params(self):
        ''' Reset episode params
        '''
        self.ep_reward, self.ep_step = 0, 0

    def init(self):
        self.curr_obs, self.curr_info = self.env.reset(seed = self.seed)
        return self.curr_obs, self.curr_info
    
    def run(self, policy, *args, **kwargs):
        dataserver, logger = kwargs['dataserver'], kwargs['logger']
        exps = []
        run_step, run_episode = 0, 0 # local run step, local run episode
        while True:
            action = policy.get_action(self.curr_obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            interact_transition = {'interactor_id': self.id, 'state': self.curr_obs, 'action': action,'reward': reward, 'next_state': obs, 'done': terminated or truncated, 'info': info}
            policy_transition = policy.get_policy_transition()
            exps.append(Exp(**interact_transition, **policy_transition))
            run_step += 1
            self.curr_obs, self.curr_info = obs, info
            self.ep_reward += reward
            self.ep_step += 1
            if terminated or truncated:
                run_episode += 1
                dataserver.increase_episode() # increase episode
                global_episode = dataserver.get_episode() # get global episode
                if global_episode % self.cfg.interact_summary_fre == 0 and global_episode <= self.cfg.max_episode: 
                    logger.info(f"Interactor {self.id} finished episode {global_episode} with reward {self.ep_reward:.3f} in {self.ep_step} steps")
                    interact_summary = {'reward':self.ep_reward,'step':self.ep_step}
                    self.update_summary((global_episode, interact_summary))
                self.reset_ep_params()
                self.curr_obs, self.curr_info = self.env.reset(seed = self.seed)
                if run_episode >= self.cfg.n_sample_episodes:
                    run_episode = 0
                    break
            if run_step >= self.cfg.n_sample_steps:
                run_step = 0
                break
        self.data = {"exps": exps, "interact_summary": self.get_summary()}
    
    def get_data(self):
        output = self.data
        self.data = None # reset data
        return output
    
    def close_env(self):
        self.env.close()

class BaseVecInteractor:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.n_envs = cfg.n_workers
        self.reset_interact_outputs()
    def reset_interact_outputs(self):
        self.interact_outputs = []

class DummyVecInteractor(BaseVecInteractor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.interactors = [BaseInteractor(cfg, id = i) for i in range(self.n_envs)]

    def run(self, policy, *args, **kwargs):
        for i in range(self.n_envs):
            self.interactors[i].run(policy, *args, **kwargs)
        for i in range(self.n_envs):
            self.interact_outputs.append(self.interactors[i].get_data())
        outputs = self.interact_outputs
        self.reset_interact_outputs()
        return outputs

    def close_envs(self):
        for i in range(self.n_envs):
            self.interactors[i].close_env()

class RayVecInteractor(BaseVecInteractor):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        if not ray.is_initialized(): ray.init()
        self.interactors = [ray.remote(BaseInteractor).options(num_cpus=0).remote(cfg, id = i) for i in range(self.n_envs)]

    def run(self, policy):
        tasks = []
        for i in range(self.n_envs):
            tasks.append(self.interactors[i].run.remote(policy))
        ready_results, _ = ray.wait(tasks, num_returns = self.n_envs, timeout = 10)
        for result in ready_results:
            id = ready_results.index(result)
            self.interact_outputs.append(ray.get(self.interactors[id].get_data.remote())) 
        outputs = self.interact_outputs
        self.reset_interact_outputs()
        return outputs
    
    def close_envs(self):
        for i in range(self.n_envs):
            ray.get(self.interactors[i].close_env.remote())

if __name__ == "__main__":
    import ray
    import time
    import random
    class Config:
        def __init__(self) -> None:
            self.n_workers = 10
            self.env_id = 'CartPole-v1'
            self.seed = 0
            self.n_sample_steps = 1000
            self.n_sample_episodes = float("inf") 
    class Policy:
        def __init__(self) -> None:
            pass
        def get_action(self,obs):
            return random.randint(0,1)
    cfg = Config()
    policy = Policy()
    env = DummyVecInteractor(cfg)
    # env = RayVecEnv(cfg)
    s_t = time.time()
    env.step(policy)
    print(time.time() - s_t)
    
  



