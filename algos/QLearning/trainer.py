#!/usr/bin/env python
# coding=utf-8
'''
Author: John
Email: johnjim0816@gmail.com
Date: 2020-09-11 23:03:00
LastEditor: John
LastEditTime: 2022-11-28 15:46:19
Discription: 
Environment: 
'''

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
        ep_reward = 0  # 一轮的累计奖励 
        ep_step = 0
        state = env.reset(seed = cfg.seed)  # 重置环境并获取初始状态
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # 采样动作 
            if cfg.new_step_api:
                next_state, reward, terminated, truncated , info = env.step(action)  # 更新环境并返回新状态、奖励、终止状态、截断标志和其他信息（使用 OpenAI Gym 的 new_step_api）
            else:
                next_state, reward, terminated, info = env.step(action)  # 更新环境并返回新状态、奖励、终止状态和其他信息（使用 OpenAI Gym 的 old_step_api） 
            agent.update(state, action, reward, next_state, terminated)  # 更新 agent
            state = next_state  # 更新状态 
            ep_reward += reward  # 增加奖励 
            if terminated:
                break
        res = {'ep_reward':ep_reward,'ep_step':ep_step}
        return agent,res
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
        ep_reward = 0  # 一轮的累计奖励 
        ep_step = 0
        state = env.reset(seed = cfg.seed)  # 重置环境并获取初始状态 
        for _ in range(cfg.max_steps):
            if cfg.render:
                env.render()
            ep_step += 1
            action = agent.predict_action(state)  # 预测动作 
            if cfg.new_step_api:
                next_state, reward, terminated, truncated , info = env.step(action)   # 更新环境并返回新状态、奖励、终止状态、截断标志和其他信息（使用 OpenAI Gym 的 new_step_api）
            else:
                next_state, reward, terminated, info = env.step(action) # 更新环境并返回新状态、奖励、终止状态和其他信息（使用 OpenAI Gym 的 old_step_api）
            state = next_state  # 更新状态 
            ep_reward += reward  # 增加奖励
            if terminated:
                break
        res = {'ep_reward':ep_reward,'ep_step':ep_step}
        return agent,res
   

        
    
