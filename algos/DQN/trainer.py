#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2022-11-22 23:19:20
LastEditor: Guoshicheng
LastEditTime: 2023-04-05 01:15:19
Discription: 
'''
import torch.multiprocessing as mp
import ray
from common.utils import all_seed


class Trainer:
    def __init__(self) -> None:
        pass
    def train_one_episode(self, env, agent, cfg): 
        ep_reward = 0  # 每回合的reward之和
        ep_step = 0 # 每回合的step之和
        state = env.reset(seed = cfg.seed)  # 重置环境并返回初始状态
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # 采样动作
            next_state, reward, terminated, truncated , info = env.step(action)  # 更新环境并返回转移
            agent.memory.push(state, action, reward, next_state, terminated)  # 存储样本(转移)
            agent.update()  # 更新智能体
            state = next_state  # 更新下一个状态
            ep_reward += reward   
            if terminated:
                break
        res = {'ep_reward':ep_reward,'ep_step':ep_step}
        return agent,res
    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  
        ep_step = 0
        ep_frames = []
        state = env.reset(seed = cfg.seed)  
        for _ in range(cfg.max_steps):
            ep_step += 1
            if cfg.render and cfg.render_mode == 'rgb_array': # 用于可视化
                frame = env.render()[0]
                ep_frames.append(frame)
            action = agent.predict_action(state) # 预测动作
            next_state, reward, terminated, truncated , info = env.step(action)  
            state = next_state  
            ep_reward += reward  
            if terminated:
                break
        res = {'ep_reward':ep_reward,'ep_step':ep_step,'ep_frames':ep_frames}
        return agent,res
    
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
                self.local_agent.memory.push(state, action, reward,next_state, terminated)
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
  

@ray.remote
class WorkerRay:
    def __init__(self,cfg,worker_id,share_agent,env,local_agent, global_r_que, global_data = None):
        self.mode = cfg.mode
        self.worker_id = worker_id
        self.global_data_objectRef = global_data
        self.global_ep = ray.get(self.global_data_objectRef.add_read_episode.remote())
        self.global_best_reward = ray.get(self.global_data_objectRef.read_best_reward.remote())
        self.global_r_que = global_r_que
        self.cfg = cfg

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
        while self.global_ep <= (self.train_eps):
            state = self.env.reset(seed = self.worker_seed)
            ep_r = 0 # reward per episode
            ep_step = 0
            while True:
                ep_step += 1
                action = self.local_agent.sample_action(state)
                next_state, reward, terminated, truncated, info = self.env.step(action)
                self.local_agent.memory.push(state, action, reward, next_state, terminated)
                # get share_agent parameters
                share_agent_policy_net, share_agent_optimizer = ray.get(self.share_agent.get_parameters.remote())
                # update share_agent
                share_agent_policy_net, share_agent_optimizer = self.local_agent.update_ray(share_agent_policy_net, share_agent_optimizer)
                # return share_agent to ShareAent
                ray.get(self.share_agent.receive_parameters.remote(share_agent_policy_net, share_agent_optimizer))
                state = next_state
                ep_r += reward
                if terminated or ep_step >= self.max_steps:
                    print(f"Worker {self.worker_id} finished episode {self.global_ep} with reward {ep_r:.3f}")
                    # record each episode and its corresponding reward in the form of a dictionary
                    self.global_r_que.put({self.global_ep:ep_r})
                    # add one to global_ep
                    self.global_ep = ray.get(self.global_data_objectRef.add_read_episode.remote())
                    break
            if (self.global_ep) % self.eval_eps == 0:
                mean_eval_reward = self.evaluate()
                if mean_eval_reward > ray.get(self.global_data_objectRef.read_best_reward.remote()):
                    ray.get(self.global_data_objectRef.set_best_reward.remote(mean_eval_reward))
                    ray.get(self.share_agent.save_model.remote(self.model_dir))
                    print(f"Worker {self.worker_id} saved model with current best eval reward {mean_eval_reward:.3f}")
    def test(self):
        while self.global_ep <= self.test_eps:
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
                    print("Worker {} finished episode {} with reward {}".format(self.worker_id,self.global_ep,ep_r))
                    self.global_r_que.put({self.global_ep:ep_r})
                    self.global_ep = ray.get(self.global_data_objectRef.add_read_episode.remote())
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
        # print(self.mode)
        if self.mode == 'train':
            self.train()
        elif self.mode == 'test':
            self.test()