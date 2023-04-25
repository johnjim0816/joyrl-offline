#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2022-12-17 19:56:24
Discription: 
Environment: 
'''
import torch.multiprocessing as mp
from common.utils import all_seed
import numpy as np

class Trainer:
    def __init__(self) -> None:
        pass
    def train_one_episode(self, env, agent, cfg): 
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset(seed = cfg.seed)  # reset and obtain initial state
        state_norm = Normalization(cfg.obs_space.shape[0])
        reward_scaling = RewardScaling(shape=1, gamma=cfg.gamma)
        state = state_norm(state)
        reward_scaling.reset()
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # sample action
            if cfg.new_step_api:
                next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            else:
                next_state, reward, terminated, info = env.step(action)  # update env and return transitions under old_step_api of OpenAI Gym
            next_state = state_norm(next_state)
            reward = float(reward_scaling(reward))
            agent.memory.push((state, action, reward, terminated, agent.probs, agent.log_probs))  # store transitions
            agent.update()  # update agent
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        res = {'ep_reward':ep_reward,'ep_step':ep_step}
        return agent,res
    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0
        ep_frames = []
        state = env.reset(seed = cfg.seed)  # reset and obtain initial state
        state_norm = Normalization(cfg.obs_space.shape[0])
        state = state_norm(state, update=False)
        for _ in range(cfg.max_steps):
            if cfg.render and cfg.render_mode == 'rgb_array':
                # env.render()
                frame = env.render()[0]
                ep_frames.append(frame)
            elif cfg.render and cfg.render_mode != 'rgb_array':
                env.render()
            ep_step += 1
            action = agent.predict_action(state)  # sample action
            if cfg.new_step_api:
                next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            else:
                next_state, reward, terminated, info = env.step(action) # update env and return transitions under old_step_api of OpenAI Gym
            next_state = state_norm(next_state)
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        res = {'ep_reward': ep_reward, 'ep_step': ep_step, 'ep_frames': ep_frames}
        return agent,res
    def collect_one_episode(self, env, agent, cfg):
        # dict of arrays
        collected = False
        ep_memory = {'state': [], 'action': [], 'next_state': [], 'reward': [], 'terminated': []}
        while not collected:
            trajectories = 0
            state = env.reset()
            ep_memory = {'state': [], 'action': [], 'next_state': [], 'reward': [], 'terminated': []}
            ep_reward = 0
            while trajectories < cfg.max_steps:
                action = agent.sample_action(state)
                new_state, reward, terminated, truncated, info = env.step(
                    action)
                ep_reward += reward
                ep_memory['state'].append(state)
                ep_memory['action'].append(action)
                ep_memory['next_state'].append(new_state)
                ep_memory['reward'].append(reward)
                ep_memory['terminated'].append(terminated)
                state = new_state
                trajectories += 1
                if terminated:
                    if ep_reward >= cfg.min_reward:
                        collected = True
                    break
        return ep_reward, ep_memory['state'], ep_memory['action'], ep_memory['next_state'], ep_memory['reward'], ep_memory['terminated']
   

        
class Worker(mp.Process):
    def __init__(self,cfg,worker_id,share_agent,env,local_agent, global_ep = None,global_r_que = None,global_best_reward = None):
        super(Worker,self).__init__()
        self.mode = cfg.mode
        self.worker_id = worker_id
        self.global_ep = global_ep
        self.global_r_que = global_r_que
        self.global_best_reward = global_best_reward

        self.share_agent = share_agent
        self.local_agent = local_agent
        self.env = env 
        self.seed = cfg.seed
        self.worker_seed = cfg.seed + worker_id
        self.train_eps = cfg.train_eps
        self.test_eps = cfg.test_eps
        self.max_steps = cfg.max_steps
        self.eval_eps = cfg.eval_eps
        self.model_dir = cfg.model_dir
    def train(self):
        while self.global_ep.value <= self.train_eps:
            state = self.env.reset(seed = self.worker_seed)
            ep_r = 0 # reward per episode
            ep_step = 0
            while True:
                ep_step += 1
                action = self.local_agent.sample_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.local_agent.memory.push((state, action, reward, terminated, self.local_agent.probs, self.local_agent.log_probs))
                self.local_agent.update(share_agent=self.share_agent)
                state = next_state
                ep_r += reward
                if terminated or ep_step >= self.max_steps:
                    print(f"Worker {self.worker_id} finished episode {self.global_ep.value} with reward {ep_r:.3f}")
                    with self.global_ep.get_lock():
                        self.global_ep.value += 1
                    self.global_r_que.put(ep_r)
                    break
            if (self.global_ep.value+1) % self.eval_eps == 0:
                mean_eval_reward = self.evaluate()
                if mean_eval_reward > self.global_best_reward.value:
                    self.global_best_reward.value = mean_eval_reward
                    self.share_agent.save_model(self.model_dir)
                    print(f"Worker {self.worker_id} saved model with current best eval reward {mean_eval_reward:.3f}")
        self.global_r_que.put(None)
    def test(self):
        while self.global_ep.value <= self.test_eps:
            state = self.env.reset(seed = self.worker_seed)
            ep_r = 0 # reward per episode
            ep_step = 0
            while True:
                ep_step += 1
                action = self.local_agent.predict_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = next_state
                ep_r += reward
                if terminated or ep_step >= self.max_steps:
                    print("Worker {} finished episode {} with reward {}".format(self.worker_id,self.global_ep.value,ep_r))
                    with self.global_ep.get_lock():
                        self.global_ep.value += 1
                    self.global_r_que.put(ep_r)
                    break
        
    def evaluate(self):
        sum_eval_reward = 0
        for _ in range(self.eval_eps):
            state = self.env.reset(seed = self.worker_seed)
            ep_r = 0 # reward per episode
            ep_step = 0
            while True:
                ep_step += 1
                action = self.local_agent.predict_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = next_state
                ep_r += reward
                if terminated or ep_step >= self.max_steps:
                    break
            sum_eval_reward += ep_r
        mean_eval_reward = sum_eval_reward / self.eval_eps
        return mean_eval_reward
    
    def run(self):
        all_seed(self.seed)
        print("worker {} started".format(self.worker_id))
        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()



class RunningMeanStd_s:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
    
    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)



class RunningMeanStd_r:
    def __init__(self, shape):
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
    
    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd_s(shape=shape)
    
    def __call__(self, x, update=True):
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x



class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape
        self.gamma = gamma
        self.running_ms = RunningMeanStd_r(shape=self.shape)
        self.R = np.zeros(self.shape)
    
    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)
        return x
    
    def reset(self):
        self.R = np.zeros(self.shape)