#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-03-10 23:35:45
LastEditor: JiangJi
LastEditTime: 2023-04-05 01:20:06
Discription: 
'''
class Trainer:
    def __init__(self) -> None:
        pass
    def train_one_episode(self, env, agent, cfg): 
        '''
        训练一回合
        Args:
            env (gym): 输入的env实例
            agent (class): 输入的agent实例
            cfg (class): 超参数配置实例
        Returns:
            agent (class): 更新一回合参数后的agent实例
            res (dict): 更新一回合后的总奖励值及更新的step总数
        '''
        ep_reward = 0  # 一回合的累计奖励 
        ep_step = 0
        state = env.reset(seed = cfg.seed)  # 重置环境并获取初始状态,即开始新的回合
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state) # 采样动作 
            if cfg.new_step_api:
                next_state, reward, terminated, truncated , info = env.step(action)  # 更新环境并返回新状态、奖励、终止状态、截断标志和其他信息（使用 OpenAI Gym 的 new_step_api）
            else:
                next_state, reward, terminated, info = env.step(action)  # 更新环境并返回新状态、奖励、终止状态、截断标志和其他信息（使用 OpenAI Gym 的 new_step_api）
            next_action =  agent.sample_action(next_state)
            agent.update(state, action, reward, next_state, next_action,terminated)  # 更新 agent
            state = next_state   # 更新状态 
            action = next_action # 更新动作
            ep_reward += reward  # 累积奖励
            if terminated:
                break
        res = {'ep_reward':ep_reward,'ep_step':ep_step}
        return agent,res
    
    def test_one_episode(self, env, agent, cfg):
        '''
        测试一回合
        Args:
            env (gym): 输入的env实例
            agent (class): 输入的agent实例
            cfg (class): 超参数配置实例
        Returns:
            agent (class): 执行完一回合后的agent实例
            res (dict): 执行一回合后的总奖励值及预测的step总数
        '''
        ep_reward = 0  # 一回合的累计奖励 
        ep_step = 0
        state = env.reset(seed = cfg.seed)   # 重置环境并获取初始状态,即开始新的回合
        for _ in range(cfg.max_steps):
            if cfg.render:
                env.render()
            ep_step += 1
            action = agent.predict_action(state)  # 预测动作 
            if cfg.new_step_api:
                next_state, reward, terminated, truncated , info = env.step(action)  # 更新环境并返回新状态、奖励、终止状态、截断标志和其他信息（使用 OpenAI Gym 的 new_step_api）
            else:
                next_state, reward, terminated, info = env.step(action) # 更新环境并返回新状态、奖励、终止状态、截断标志和其他信息（使用 OpenAI Gym 的 new_step_api）
            state = next_state   # 更新状态 
            ep_reward += reward  # 增加奖励
            if terminated:
                break
        res = {'ep_reward':ep_reward,'ep_step':ep_step}
        return agent,res