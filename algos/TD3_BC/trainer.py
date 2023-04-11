#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-02-21 20:32:11
LastEditor: JiangJi
LastEditTime: 2023-02-23 22:05:19
Discription: 
'''
import numpy as np

class Trainer:
    def __init__(self) -> None:
        pass


    def train_one_episode(self, env, agent, cfg): 
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset(seed = cfg.seed)  # reset and obtain initial state
        train_iterations = cfg.train_iterations
        batch_size = cfg.batch_size
        expert_states = agent.expert_states; expert_actions = agent.expert_actions
        expert_next_states = agent.expert_next_states; expert_rewards = agent.expert_rewards
        expert_terminals = agent.expert_terminals

        for i in range(train_iterations):
            sample_indices = np.random.randint(low=0,
                                            high=expert_states.shape[0],
                                            size=batch_size)
            agent.update(expert_states[sample_indices], expert_actions[sample_indices], expert_next_states[sample_indices], \
                expert_rewards[sample_indices], expert_terminals[sample_indices])
            agent,res = self.test_one_episode(env, agent, cfg)
            print (f"iter: {i + 1}/{cfg.train_iterations}, Reward: {res['ep_reward']:.3f}, Step: {res['ep_step']}")
        res = {'ep_reward':ep_reward,'ep_step':ep_step}
        return agent,res

    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset(seed = cfg.seed)  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1
            state = (np.array(state).reshape(1,-1) - agent.mean)/agent.std
            state = state[0]

            action = agent.predict_action(state)  # sample action
            next_state, reward, terminated, truncated, info = env.step(action)  # update env and return transitions under new_step_api of OpenAI Gym
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break
        res = {'ep_reward':ep_reward,'ep_step':ep_step}
        return agent,res