#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2021-12-22 10:40:05
LastEditor: JiangJi
LastEditTime: 2023-04-16 11:11:47
Discription: 
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from common.memories import ReplayBufferQue
from common.models import MLP, Critic

class Agent(object):
	def __init__(self,cfg):
		'''智能体类
        Args:array
            cfg (class): 超参数类
        '''
		self.gamma = cfg.gamma
		self.actor_lr = cfg.actor_lr
		self.critic_lr = cfg.critic_lr
		self.policy_noise = cfg.policy_noise # noise added to target policy during critic update
		self.noise_clip = cfg.noise_clip # range to clip target policy noise
		self.expl_noise = cfg.expl_noise # std of Gaussian exploration noise
		self.policy_freq = cfg.policy_freq # policy update frequency
		self.batch_size =  cfg.batch_size 
		self.tau = cfg.tau
		self.sample_count = 0
		self.explore_steps = cfg.explore_steps # exploration steps before training
		self.device = torch.device(cfg.device)
		self.n_actions = cfg.n_actions
		self.action_space = cfg.action_space
		self.actor_input_dim = cfg.n_states
		self.actor_output_dim = cfg.n_actions
		self.critic_input_dim = cfg.n_states + cfg.n_actions
		self.critic_output_dim = 1

		self.action_scale = torch.tensor((self.action_space.high - self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0) # 动作空间放缩
		self.action_bias = torch.tensor((self.action_space.high + self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0) # 动作空间平移
		self.actor = MLP(self.actor_input_dim, self.actor_output_dim, hidden_dim = cfg.actor_hidden_dim).to(self.device)
		self.actor_target = MLP(self.actor_input_dim, self.actor_output_dim, hidden_dim = cfg.actor_hidden_dim).to(self.device)
		self.actor_target.load_state_dict(self.actor.state_dict())

		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.actor_lr)

		self.critic_1 = Critic(self.critic_input_dim, self.critic_output_dim, hidden_dim = cfg.critic_hidden_dim).to(self.device)
		self.critic_2 = Critic(self.critic_input_dim, self.critic_output_dim, hidden_dim = cfg.critic_hidden_dim).to(self.device)
		self.critic_1_target = Critic(self.critic_input_dim, self.critic_output_dim, hidden_dim = cfg.critic_hidden_dim).to(self.device)
		self.critic_2_target = Critic(self.critic_input_dim, self.critic_output_dim, hidden_dim = cfg.critic_hidden_dim).to(self.device)
		self.critic_1_target.load_state_dict(self.critic_1.state_dict())
		self.critic_2_target.load_state_dict(self.critic_2.state_dict())
		
		self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr = self.critic_lr)
		self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr = self.critic_lr)

		self.alpha = cfg.alpha
		self.lmbda = cfg.lmbda

		self.expert_path = f"{cfg.expert_path}" # 专家数据路径
		with open(self.expert_path, 'rb') as f:
			expert_data = pickle.load(f)
		self.expert_states = np.array(expert_data['states']) 
		self.expert_actions = np.array(expert_data['actions'])
		self.expert_next_states = np.array(expert_data['next_states']) 
		self.expert_rewards = np.array(expert_data['rewards'])
		self.expert_terminals = np.array(expert_data['terminals'])

		if cfg.normalize:
			self.mean, self.std, self.expert_states, self.expert_next_states = self.normalize_states(self.expert_states, self.expert_next_states) # 归一化专家状态
		else:
			self.mean, self.std = 0, 1

	def normalize_states(self, expert_states, expert_next_states, eps = 1e-3):
		'''归一化 state
		Args:
			expert_states (array): 专家状态
			expert_next_states (array): 专家下一个状态
			eps (float, optional): 防止 std 为 0 的参数. Defaults to 1e-3.
		Returns:
			mean (float): 均值
			std (float): 方差
			expert_states (array): 归一化后的专家状态
			expert_next_states (array): 归一化后的专家下一个状态
		'''
		mean = expert_states.mean(0,keepdims=True)
		std = expert_states.std(0,keepdims=True) + eps
		print (mean, std)
		expert_states = (expert_states - mean)/std
		expert_next_states = (expert_next_states - mean)/std
		return mean, std, expert_states, expert_next_states 

	def sample_action(self, state):
		'''采样动作
		Args:
			state (array): 状态
		Returns:
			action (array): 动作
		'''
		self.sample_count += 1
		if self.sample_count < self.explore_steps:
			return self.action_space.sample()
		else:
			state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
			action = torch.tanh(self.actor(state))
			action = self.action_scale * action + self.action_bias # 对动作进行放缩和平移
			action = action.detach().cpu().numpy()[0]
			action_noise = np.random.normal(0, self.action_scale.cpu().numpy()[0] * self.expl_noise, size=self.n_actions) # 高斯噪音
			action = (action + action_noise).clip(self.action_space.low, self.action_space.high)
			return action

	@torch.no_grad()
	def predict_action(self, state):
		'''预测动作
        Args:
            state (array): 状态
        Returns:
            action (array): 动作
        '''
		state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
		action = torch.tanh(self.actor(state))
		action = self.action_scale * action + self.action_bias
		return action.detach().cpu().numpy()[0]

	def update(self, state, action, next_state, reward, done):
		'''更新网络
		Args:
			state (array): 状态
			action (array): 动作
			next_state (array): 下一状态
			reward (array): 奖励
			done (array): 是否结束
		'''
		# convert to tensor
		state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32)
		action = torch.tensor(np.array(action), device=self.device, dtype=torch.float32)
		next_state = torch.tensor(np.array(next_state), device=self.device, dtype=torch.float32)
		reward = torch.tensor(reward, device=self.device, dtype=torch.float32).unsqueeze(1)
		done = torch.tensor(done, device=self.device, dtype=torch.float32).unsqueeze(1)

		# update critic
		noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
		next_action = ((torch.tanh(self.actor_target(next_state)) + noise) * self.action_scale + self.action_bias).clamp(-self.action_scale+self.action_bias, self.action_scale+self.action_bias)
		next_sa = torch.cat([next_state, next_action], 1) # shape:[train_batch_size,n_states+n_actions]
		target_q1, target_q2 = self.critic_1_target(next_sa).detach(), self.critic_2_target(next_sa).detach()
		target_q = torch.min(target_q1, target_q2) # shape:[train_batch_size,n_actions] 
		target_q = reward + self.gamma * target_q * (1 - done) # 目标 Q 值
		sa = torch.cat([state, action], 1)
		current_q1, current_q2 = self.critic_1(sa), self.critic_2(sa) # 当前 Q 值
		# compute critic loss
		critic_1_loss = F.mse_loss(current_q1, target_q)
		critic_2_loss = F.mse_loss(current_q2, target_q)
		self.critic_1_optimizer.zero_grad()
		critic_1_loss.backward()
		self.critic_1_optimizer.step()
		self.critic_2_optimizer.zero_grad()
		critic_2_loss.backward()
		self.critic_2_optimizer.step()
		# Delayed policy updates
		if self.sample_count % self.policy_freq == 0:
			# compute actor loss

			probs = torch.tanh(self.actor(state))
			probs_scale = self.action_scale * probs + self.action_bias # 对动作进行放缩和平移
			q_critic = self.critic_1(torch.cat([state, probs_scale], 1))
			# with torch.no_grad():
			lmbda = self.alpha/q_critic.abs().mean().detach() # 计算 Q 值的权重
			# TD3 + BC
			actor_loss = - lmbda * q_critic.mean() + F.mse_loss(probs_scale, action) # actor 网络更新目标为保证策略网络输出动作接近专家动作的基础上最大化动作 Q 值

			self.actor_optimizer.zero_grad()
			actor_loss.backward()
			self.actor_optimizer.step()
			## 软更新 critic 网络和 actor 网络
			for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
			for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

	def save_model(self, fpath):
		'''保存网络权重
        Args:
            fpath (str): 保存权重文件路径
        '''
		from pathlib import Path
        # create path
		Path(fpath).mkdir(parents=True, exist_ok=True)
		torch.save(self.critic_1.state_dict(), f"{fpath}/critic_1.pth")
		torch.save(self.critic_2.state_dict(), f"{fpath}/critic_2.pth")
		torch.save(self.actor.state_dict(), f"{fpath}/actor.pth")

	def load_model(self, fpath):
		'''加载网络权重
        Args:
            fpath (str): 加载权重文件路径
        '''
		critic_1_ckpt = torch.load(f"{fpath}/critic_1.pth", map_location=self.device)
		critic_2_ckpt = torch.load(f"{fpath}/critic_2.pth", map_location=self.device)
		actor_ckpt = torch.load(f"{fpath}/actor.pth", map_location=self.device)
		self.critic_1.load_state_dict(critic_1_ckpt)
		self.critic_2.load_state_dict(critic_2_ckpt)
		self.actor.load_state_dict(actor_ckpt)
		
