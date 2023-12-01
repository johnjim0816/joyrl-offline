import gymnasium as gym
import copy
from typing import Tuple
from algos.base.exps import Exp
from framework.message import Msg, MsgType
from config.general_config import MergedConfig

class BaseInteractor:
    ''' Interactor for gym env to support sample n-steps or n-episodes traning data
    '''
    def __init__(self, cfg: MergedConfig, id = 0, env = None, policy = None, *args, **kwargs) -> None:
        self.cfg = cfg 
        self.id = id
        self.policy = policy
        self.env = env
        self.seed = self.cfg.seed + self.id
        self.exps = [] # reset experiences
        self.summary = [] # reset summary
        self.ep_reward, self.ep_step = 0, 0 # reset params per episode
        self.curr_obs, self.curr_info = self.env.reset(seed = self.seed) # reset env
    
    def run(self, *args, **kwargs):
        collector = kwargs['collector']
        recorder = kwargs['recorder']
        model_mgr = kwargs['model_mgr']
        model_params = model_mgr.pub_msg(Msg(type = MsgType.MODEL_MGR_GET_MODEL_PARAMS)) # get model params
        self.policy.put_model_params(model_params)
        self._sample_data(*args, **kwargs)
        collector.pub_msg(Msg(type = MsgType.COLLECTOR_PUT_EXPS, data = self.exps)) # put exps to collector
        self.exps = [] # reset exps
        if len(self.summary) > 0:
            recorder.pub_msg(Msg(type = MsgType.RECORDER_PUT_INTERACT_SUMMARY, data = self.summary)) # put summary to stats recorder
            self.summary = [] # reset summary

    def _sample_data(self,*args, **kwargs):
        tracker = kwargs['tracker']
        logger = kwargs['logger']
        run_step, run_episode = 0, 0 # local run step, local run episode
        while True:
            action = self.policy.get_action(self.curr_obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            interact_transition = {'interactor_id': self.id, 'state': self.curr_obs, 'action': action,'reward': reward, 'next_state': obs, 'done': terminated or truncated, 'info': info}
            policy_transition = self.policy.get_policy_transition()
            self.exps.append(Exp(**interact_transition, **policy_transition))
            run_step += 1
            self.curr_obs, self.curr_info = obs, info
            self.ep_reward += reward
            self.ep_step += 1
            if terminated or truncated or self.ep_step >= self.cfg.max_step:
                run_episode += 1
                tracker.pub_msg(Msg(MsgType.TRACKER_INCREASE_EPISODE))
                global_episode = tracker.pub_msg(Msg(MsgType.TRACKER_GET_EPISODE))
                if global_episode % self.cfg.interact_summary_fre == 0 and global_episode <= self.cfg.max_episode: 
                    logger.info(f"Interactor {self.id} finished episode {global_episode} with reward {self.ep_reward:.3f} in {self.ep_step} steps")
                    interact_summary = {'reward':self.ep_reward,'step':self.ep_step}
                    self.summary.append((global_episode, interact_summary))
                self.ep_reward, self.ep_step = 0, 0 # reset params per episode
                self.curr_obs, self.curr_info = self.env.reset(seed = self.seed) # reset env
                if run_episode >= self.cfg.n_sample_episodes:
                    run_episode = 0
                    break
            if run_step >= self.cfg.n_sample_steps:
                run_step = 0
                break
    

class BaseWorker:
    def __init__(self, cfg: MergedConfig, policy = None, *args, **kwargs) -> None:
        self.cfg = cfg
        self.n_envs = cfg.n_workers

class DummyWorker(BaseWorker):
    def __init__(self, cfg: MergedConfig, env = None, policy = None, *args, **kwargs) -> None:
        super().__init__(cfg, env = env, policy = policy, *args, **kwargs)
        self.interactors = [BaseInteractor(cfg, id = i, env = copy.deepcopy(env), policy = copy.deepcopy(policy), *args, **kwargs) for i in range(self.n_envs)]

    def run(self, *args, **kwargs):
        for i in range(self.n_envs):
            self.interactors[i].run(*args, **kwargs)

class RayWorker(BaseWorker):
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
    env = DummyWorker(cfg)
    # env = RayVecEnv(cfg)
    s_t = time.time()
    env.step(policy)
    print(time.time() - s_t)
    
  



