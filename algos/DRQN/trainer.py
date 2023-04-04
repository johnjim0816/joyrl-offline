#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2023-04-05 01:17:52
Discription: 
Environment: 
'''
import torch
from common.memories import ReplayBuffer

class Trainer:
    def __init__(self) -> None:
        pass
    def train_one_episode(self, env, agent, cfg): 
        ep_reward = 0  # reward per episode
        ep_step = 0
        episode_record = ReplayBuffer(cfg.buffer_size) 
        h,c = torch.zeros([1, 1, cfg.hidden_dim]), torch.zeros([1, 1, cfg.hidden_dim])
        state = env.reset(seed = cfg.seed)  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1
            action, h, c = agent.sample_action(state, h, c)  # sample action
            if cfg.new_step_api:
                next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            else:
                next_state, reward, terminated, info = env.step(action)  # update env and return transitions under old_step_api of OpenAI Gym

            episode_record.push(state, action, reward / 100.0, next_state, terminated) ## needed to divide by 100.0 
            agent.update()  # update agent
            state = next_state  # update next state for env
            ep_reward += reward  
            if terminated:
                break
        agent.epsilon = max(agent.epsilon_end , agent.epsilon * cfg.epsilon_decay)
        agent.memory.push(episode_record)
        res = {'ep_reward':ep_reward,'ep_step':ep_step}
        return agent,res
    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset(seed = cfg.seed)  # reset and obtain initial state
        h,c = torch.zeros([1, 1, cfg.hidden_dim]), torch.zeros([1, 1, cfg.hidden_dim])
        for _ in range(cfg.max_steps):
            if cfg.render:
                env.render()
            ep_step += 1
            action, h, c = agent.predict_action(state, h, c)  # sample action
            if cfg.new_step_api:
                next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            else:
                next_state, reward, terminated, info = env.step(action) # update env and return transitions under old_step_api of OpenAI Gym
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        res = {'ep_reward':ep_reward,'ep_step':ep_step}
        return agent,res
   

        
    
