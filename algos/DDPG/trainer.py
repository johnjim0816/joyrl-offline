#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-02-21 20:32:11
LastEditor: JiangJi
LastEditTime: 2023-05-25 20:19:46
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

