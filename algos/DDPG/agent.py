#!/usr/bin/env python
# coding=utf-8
'''
@Author: John
@Email: johnjim0816@gmail.com
@Date: 2020-06-09 20:25:52
@LastEditor: John
LastEditTime: 2022-12-06 22:50:45
@Discription:
@Environment: python 3.7.7
'''
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from common.memories import ReplayBufferQue


class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        '''
        actor 模型的结构定义
        Args:
            n_states (int): 输入状态的维度
            n_actions (int): 可执行动作的数量
            hidden_dim (int): 隐含层数量
            init_w (float, optional): 均匀分布初始化权重的范围
        '''
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        return x


class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        '''
        critic 模型的结构定义
        Args:
            n_states (int): 输入状态的维度
            n_actions (int): 可执行动作的数量
            hidden_dim (int): 隐含层数量
            init_w (float, optional): 均匀分布初始化权重的范围
        '''
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class OUNoise(object):
    '''
    构造 Ornstein–Uhlenbeck 噪声的类
    https://djalil.chafai.net/docs/M2/history-brownian-motion/Uhlenbeck%20&%20Ornstein%20-%201930.pdf
    '''

    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        '''
        初始化输入参数
        Args:
            action_space (gym.spaces.box.Box): env 中的 action_space
            mu (float, optional): 噪声均值
            theta (float, optional): 系统对噪声的扰动程度，theta 越大，噪声扰动越小
            max_sigma (float, optional): 最大 sigma，用于更新衰变 sigma 值
            min_sigma (float, optional): 最小 sigma，用于更新衰变 sigma 值
            decay_period (int, optional): 衰变周期
        '''
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.n_actions = action_space.shape[0]  # env环境中可执行动作的数量，因是连续动作，所以一般为1
        self.low = action_space.low  # env环境中动作取值的最小值
        self.high = action_space.high  # env环境中动作取值的最大值
        self.reset()

    def reset(self):
        '''
        重置噪声
        '''
        self.obs = np.ones(self.n_actions) * self.mu  # reset the noise

    def evolve_obs(self):
        '''
        更新噪声
        Returns:
            返回更新后的噪声值
        '''
        x = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions)  # Ornstein–Uhlenbeck process
        self.obs = x + dx
        return self.obs

    def get_action(self, action, t=0):
        '''
        根据输入的动作，输出加入 OU 噪声后的动作
        Args:
            action (np.ndarray[float]): 输入的动作值
            t (int, optional): 当前环境已执行的帧数

        Returns:
            action (np.ndarray[float]): 返回加入 OU 噪声后的动作
        '''
        ou_obs = self.evolve_obs()
        ## 根据env进程（t），通过设定的衰变周期（decay_period），进行更新衰变的sigma值
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_obs, self.low, self.high)  # add noise to action


class Agent:
    def __init__(self, cfg):
        '''
        构建智能体
        Args:
            cfg (class): 超参数类 AlgoConfig
        '''
        self.n_states = cfg.n_states
        self.n_actions = cfg.n_actions
        self.action_space = cfg.action_space  # env 中的 action_space
        self.ou_noise = OUNoise(self.action_space)  # 实例化 构造 Ornstein–Uhlenbeck 噪声的类
        self.batch_size = cfg.batch_size
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.sample_count = 0  # 记录采样动作的次数
        self.update_flag = False  # 标记是否更新网络
        self.device = torch.device(cfg.device)
        self.critic = Critic(self.n_states, self.n_actions, hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.target_critic = Critic(self.n_states, self.n_actions, hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.actor = Actor(self.n_states, self.n_actions, hidden_dim=cfg.actor_hidden_dim).to(self.device)
        self.target_actor = Actor(self.n_states, self.n_actions, hidden_dim=cfg.actor_hidden_dim).to(self.device).to(
            self.device)
        ## 将 critc 网络的参数赋值给target critic 网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        ## 将 actor 网络的参数赋值给target actor 网络
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.memory = ReplayBufferQue(cfg.buffer_size)

    def sample_action(self, state):
        '''
        根据输入的状态采样动作
        Args:
            state (np.ndarray): 输入的状态

        Returns:
            action (np.ndarray[float]): 根据状态采样后的动作
        '''
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        action_tanh = self.actor(state)  # action_tanh is in [-1, 1]
        # convert action_tanh to action in the original action space
        action_scale = torch.FloatTensor((self.action_space.high - self.action_space.low) / 2.).to(self.device)
        action_bias = torch.FloatTensor((self.action_space.high + self.action_space.low) / 2.).to(self.device)
        action = action_scale * action_tanh + action_bias
        action = action.cpu().detach().numpy()[0]
        # add noise to action
        action = self.ou_noise.get_action(action, self.sample_count)
        return action

    @torch.no_grad()
    def predict_action(self, state):
        '''
        根据输入的状态预测下一步的动作
        Args:
            state (np.ndarray):  输入的状态

        Returns:
            action (np.ndarray[float]): 根据状态采样后的动作
        '''
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        action_tanh = self.actor(state)  # action_tanh is in [-1, 1]
        # convert action_tanh to action in the original action space
        action_scale = torch.FloatTensor((self.action_space.high - self.action_space.low) / 2.).to(self.device)
        action_bias = torch.FloatTensor((self.action_space.high + self.action_space.low) / 2.).to(self.device)
        action = action_scale * action_tanh + action_bias
        action = action.cpu().detach().numpy()[0]
        return action

    def update(self):
        ## 当经验回放池中的数量小于 batch_size 时，直接返回不更新
        if len(self.memory) < self.batch_size:  # when memory size is less than batch size, return
            return
        else:
            if not self.update_flag:
                print("Begin to update!")
                self.update_flag = True
        ## 从经验回放池中采样 batch_size 个样本
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        ## 将状态、动作等 array 转为 tensor
        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)
        ## 输入状态及通过 actor 网络 根据该状态输出的动作，计算 critic 网络输出的价值，类似于DQN中的q-value
        policy_loss = self.critic(state, self.actor(state))
        ## 计算均值作为 critic 网络的损失
        policy_loss = -policy_loss.mean()
        ## 根据下一个 time step 的状态用 target_actor 网络输出目标动作
        next_action = self.target_actor(next_state)
        ## 输入下一个 time step 的状态及目标动作，计算 target_critc 网络输出的目标价值
        target_value = self.target_critic(next_state, next_action.detach())
        ## 根据真实奖励更新目标价值
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)
        ## 输入状态和动作，用 critic 网络计算预估的价值
        value = self.critic(state, action)
        ## 将 critic 网络输出的价值和 target_critic输出并更新后的价值通过MSE进行损失计算
        value_loss = nn.MSELoss()(value, expected_value.detach())
        ## 更新 actor 网络参数
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        ## 更新 critic 网络参数
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        ## 通过软更新的方法，缓慢更新 target critic 网络的参数
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                param.data * self.tau
            )
        ## 通过软更新的方法，缓慢更新 target actor 网络的参数
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) +
                param.data * self.tau
            )

    def save_model(self, fpath):
        '''
        保存模型
        Args:
            fpath (str): 模型存放路径
        '''
        from pathlib import Path
        # create path

        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{fpath}/actor_checkpoint.pt")

    def load_model(self, fpath):
        '''
        根据模型路径导入模型
        Args:
            fpath (str): 模型路径
        '''
        actor_ckpt = torch.load(f"{fpath}/actor_checkpoint.pt", map_location=self.device)
        self.actor.load_state_dict(actor_ckpt)
