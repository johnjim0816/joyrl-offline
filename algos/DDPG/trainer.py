#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-02-21 20:32:11
LastEditor: JiangJi
LastEditTime: 2023-04-16 17:24:08
Discription:
'''
import torch.multiprocessing as mp
from utils.utils import all_seed

class Trainer:
    def __init__(self) -> None:
        pass

    def train_one_episode(self, env, agent, cfg):
        '''
        更新一轮参数
        Args:
            env (gym): 输入的env实例
            agent (class): 输入的agent实例
            cfg (class): 超参数配置实例

        Returns:
            agent (class): 更新一轮参数后的agent实例
            res (dict): 更新一轮后的总奖励值及更新的step总数
        '''
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # sample action
            next_state, reward, terminated, truncated, info = env.step(
                action)  # update env and return transitions under new_step_api of OpenAI Gym
            agent.memory.push((state, action, reward,
                               next_state, terminated))  # save transitions
            agent.update()  # update agent
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        res = {'ep_reward': ep_reward, 'ep_step': ep_step}
        return agent, res

    def test_one_episode(self, env, agent, cfg):
        '''
        预测一轮
        Args:
            env (gym): 输入的env实例
            agent (class): 输入的agent实例
            cfg (class): 超参数配置实例

        Returns:
            agent (class): 执行完一轮后的agent实例
            res (dict): 执行一轮后的总奖励值及预测的step总数
        '''
        ep_reward = 0  # reward per episode
        ep_step = 0
        ep_frames = []
        state = env.reset(seed = cfg.seed)  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1
            if cfg.render and cfg.render_mode == 'rgb_array': # 用于可视化
                frame = env.render()[0]
                ep_frames.append(frame)
            action = agent.predict_action(state)  # sample action
            next_state, reward, terminated, truncated, info = env.step(
                action)  # update env and return transitions under new_step_api of OpenAI Gym
            state = next_state  # update next state for env
            ep_reward += reward
            if terminated:
                break
        res = {'ep_reward':ep_reward,'ep_step':ep_step,'ep_frames':ep_frames}
        return agent, res


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
                self.local_agent.memory.push((state, action, reward, next_state, terminated))
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