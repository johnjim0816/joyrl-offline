#!/usr/bin/env python
# coding=utf-8
import numpy as np
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2022-12-17 19:56:24
Discription: 
Environment: 
'''

class Trainer:
    def __init__(self) -> None:
        pass
    def sample_expert_data(self, env, agent, cfg): # , expert_episode_num
        expert_states = [] ; expert_actions = []
        for e in range(cfg.expert_episode_num):
            state = env.reset(seed = cfg.seed)
            ep_step = 0
            for _ in range(cfg.max_steps):
                ep_step += 1
                action = agent.expert_predict_action(state)  # sample action
                expert_states.append(state) ; expert_actions.append(action) 
                if cfg.new_step_api:
                    next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
                else:
                    next_state, reward, terminated, info = env.step(action)  # update env and return transitions under old_step_api of OpenAI Gym
                state = next_state  # update next state for env 
                if terminated:
                    break                
        return np.array(expert_states), np.array(expert_actions)
           
    def train_one_episode(self, env, agent, cfg): 
        expert_states, expert_actions = self.sample_expert_data(env, agent, cfg)
        train_iterations = cfg.train_iterations
        batch_size = cfg.train_batch_size
        for i in range(train_iterations):
            sample_indices = np.random.randint(low=0,
                                            high=expert_states.shape[0],
                                            size=batch_size)
            agent.update_policy(expert_states[sample_indices], expert_actions[sample_indices])
            agent,ep_reward,ep_step = self.test_one_episode(env, agent, cfg)
            print (f"iter: {i + 1}/{cfg.train_iterations}, Reward: {ep_reward:.3f}, Step: {ep_step}")
        return agent,ep_reward,ep_step
    
    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset(seed = cfg.seed)  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            if cfg.render:
                env.render()
            ep_step += 1
            action = agent.predict_action(state)  # sample action
            if cfg.new_step_api:
                next_state, reward, terminated, truncated , info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            else:
                next_state, reward, terminated, info = env.step(action) # update env and return transitions under old_step_api of OpenAI Gym
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        return agent,ep_reward,ep_step
