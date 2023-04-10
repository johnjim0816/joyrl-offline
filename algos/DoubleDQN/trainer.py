#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-02-21 20:32:11
LastEditor: JiangJi
LastEditTime: 2023-04-05 01:17:26
Discription: 
'''


class Trainer:
    '''训练类
    '''
    def __init__(self) -> None:
        pass

    def train_one_episode(self, env, agent, cfg):
        '''定义一个回合的训练
        Args:
            env(class): 环境类
            agent(class): 智能体类
            cfg(class): 超参数类
        Returns:
            agent(class):智能体类
            res(dict): 一个回合的结果 keys={'ep_reward', 'ep_step'}
                ep_reward(float): 一个回合获得的回报
                ep_step(int): 一个回合总步数
        '''
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1  # 时间步
            action = agent.sample_action(state)  # sample action
            next_state, reward, terminated, truncated, info = env.step(
                action)  # update env and return transitions under new_step_api of OpenAI Gym
            agent.memory.push((state, action, reward, next_state, terminated))  # save transitions
            agent.update()  # update agent
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:  # 回合结束
                break
        res = {'ep_reward': ep_reward, 'ep_step': ep_step}  # ep_reward:一个回合获得的回报, ep_step:一个回合总步数
        return agent, res

    def test_one_episode(self, env, agent, cfg):
        '''定义一个回合的测试
        Args:
            env(class): 环境类
            agent(class): 智能体类
            cfg(class): 超参数类
        Returns:
            agent(class):智能体类
            res(dict): 一个回合的结果 keys={'ep_reward', 'ep_step'}
                ep_reward(float): 一个回合获得的回报
                ep_step(int): 一个回合总步数
        '''
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1  # 时间步
            action = agent.predict_action(state)  # sample action
            next_state, reward, terminated, truncated, info = env.step(
                action)  # update env and return transitions under new_step_api of OpenAI Gym
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:  # 回合结束
                break
        res = {'ep_reward': ep_reward, 'ep_step': ep_step}
        return agent, res
