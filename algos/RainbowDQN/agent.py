import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random
import numpy as np
import ray
from common.memories import ReplayBufferQue, ReplayBuffer, ReplayTree
from common.optms import SharedAdam

'''
This NoisyLinear is modified from the original code from 
https://github.com/higgsfield/RL-Adventure/blob/master/5.noisy%20dqn.ipynb
'''


class NoisyLinear(nn.Module):
    '''
    定义 Noisy Networks: https://arxiv.org/pdf/1706.10295.pdf
    '''
    def __init__(self, input_dim, output_dim, std_init=0.4):
        super(NoisyLinear, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_dim, input_dim))
        ## 定义 weight epsilone, 因该参数不作为训练中可调整的参数，所以需用 register_buffer 进行参数注册
        self.register_buffer('weight_epsilon', torch.FloatTensor(output_dim, input_dim))

        self.bias_mu = nn.Parameter(torch.FloatTensor(output_dim))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_dim))
        ## 定义 bias epsilon, 因该参数不作为训练中可调整的参数，所以需用 register_buffer 进行参数注册
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_dim))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(torch.tensor(self.weight_epsilon))
            bias = self.bias_mu + self.bias_sigma.mul(torch.tensor(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        '''
        初始化参数
        '''
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        '''
        初始化噪音参数
        '''
        epsilon_in = self._scale_noise(self.input_dim)
        epsilon_out = self._scale_noise(self.output_dim)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.output_dim))

    def _scale_noise(self, size):
        '''
        计算 Factorised Gaussian noise 时使用的 real-valued function
        Args:
            size (int): 输入大小

        Returns:
            x (float): 返回的 real-valued
        '''
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x


class DistributionalNetwork(nn.Module):
    '''
    定义结合了 Dueling DQN 和 Distributional DQN 的网络
    Distributional DQN: https://arxiv.org/pdf/1707.06887.pdf
    Dueling DQN: https://arxiv.org/pdf/1511.06581.pdf
    '''
    def __init__(self, n_states, hidden_dim, n_actions, n_atoms, Vmin, Vmax):
        super(DistributionalNetwork, self).__init__()
        self.n_atoms = n_atoms  # 定义值分布时使用的离散分布中的参数，atoms 的数量，表示离散分布的柱状数量，即价值采样点
        '''Vmin,Vmax: Range of the support of rewards. Ideally, it should be [min, max], '
                             'where min and max are referred to the min/max cumulative discounted '
                             'reward obtainable in one episode. Defaults to [0, 200].'
        '''
        self.Vmin = Vmin  # 定义值分布时使用的离散分布中的参数，表示分布所代表价值的范围下界
        self.Vmax = Vmax  # 定义值分布时使用的离散分布中的参数，表示分布所代表价值的范围上界
        self.delta_z = (Vmax - Vmin) / (n_atoms - 1)  # 表示在值分布中离散分布的柱状密集程度
        self.n_actions = n_actions  # 动作数量

        self.fc1 = nn.Linear(n_states, hidden_dim)
        self.noisy_value2 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy_value3 = NoisyLinear(hidden_dim, n_atoms)
        ## 定义 Dueling DQN 中的 advantage 网络，并结合了 Noise Network
        self.noisy_advantage2 = NoisyLinear(hidden_dim, hidden_dim)  # NoisyDQN + Dueling DQN
        self.noisy_advantage3 = NoisyLinear(hidden_dim, n_actions * n_atoms)

        self.register_buffer('supports', torch.arange(Vmin, Vmax + self.delta_z, self.delta_z))
        # self.reset_parameters()

    def dist(self, x):
        '''
        计算在当前状态下的价值分布
        Args:
            x (tensor): 输入状态

        Returns:
            x (tensor): 价值分布
        '''
        x = torch.relu(self.fc1(x))

        value = F.relu(self.noisy_value2(x))
        value = self.noisy_value3(value).view(-1, 1, self.n_atoms)

        advantage = F.relu(self.noisy_advantage2(x))
        advantage = self.noisy_advantage3(advantage).view(-1, self.n_actions, self.n_atoms)

        x = value + advantage - advantage.mean(dim=1, keepdim=True)
        x = x.view(-1, self.n_actions, self.n_atoms)
        x = torch.softmax(x, dim=-1)
        return x

    def forward(self, x):
        x = self.dist(x)
        x = torch.sum(x * self.supports, dim=2)
        return x

    def reset_noise(self):
        '''
        初始化噪音参数
        '''
        self.noisy_value2.reset_noise()
        self.noisy_value3.reset_noise()

        self.noisy_advantage2.reset_noise()
        self.noisy_advantage3.reset_noise()


class Agent:
    def __init__(self, cfg) -> None:
        self.n_actions = cfg.n_actions
        self.n_atoms = cfg.n_atoms
        self.Vmin = cfg.Vmin
        self.Vmax = cfg.Vmax
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.device = torch.device(cfg.device)
        ## 用 值分布网络 定义 policy 网络
        self.policy_net = DistributionalNetwork(cfg.n_states, cfg.hidden_dim, cfg.n_actions, cfg.n_atoms, cfg.Vmin,
                                                cfg.Vmax).to(self.device)
        ## 用 值分布网络 定义 target 网络
        self.target_net = DistributionalNetwork(cfg.n_states, cfg.hidden_dim, cfg.n_actions, cfg.n_atoms, cfg.Vmin,
                                                cfg.Vmax).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        # self.memory = ReplayBuffer(cfg.buffer_size) # ReplayBufferQue(cfg.capacity)
        self.memory = ReplayTree(cfg.buffer_size)
        self.sample_count = 0  # 记录采样动作的次数

        self.n_step = cfg.n_step  # used for N-step DQN

        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update

    def sample_action(self, state):
        '''
        根据输入的状态采样动作
        Args:
            state (np.ndarray): 输入的状态

        Returns:
            action (np.ndarray[float]): 根据状态采样后的动作
        '''
        self.sample_count += 1
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                       math.exp(-1. * self.sample_count / self.epsilon_decay)
        if random.random() > self.epsilon:
            action = self.predict_action(state)
        else:
            action = random.randrange(self.n_actions)
        return action

    def predict_action(self, state):
        '''
        根据输入的状态预测下一步的动作
        Args:
            state (np.ndarray):  输入的状态

        Returns:
            action (np.ndarray[float]): 根据状态采样后的动作
        '''
        with torch.no_grad():
            state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            # print ("state", state)
            q_values = self.policy_net(state)
            action = q_values.max(1)[1].item()
            # action = q_values.argmax() // self.n_atoms
            # action = action.item()  # choose action corresponding to the maximum q value
        return action

    def update(self):
        ## 当经验回放池中的数量小于 batch_size 时，直接返回不更新
        if len(self.memory) < self.batch_size:
            return
        # states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        ## 从经验回放池中采样 batch_size 个样本
        (states, actions, rewards, next_states, dones), idxs_batch, is_weights_batch = self.memory.sample(
            self.batch_size)
        ## 将状态、动作等 array 转为 tensor
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(dim=1)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        # calculate the distribution of the next state

        with torch.no_grad():
            # next_action = self.policy_net --> DDQN  self.target_net --> DQN
            ## 用 policy 网络根据下一个状态预测动作分布，并选出期望最大的动作 a^*
            next_action = self.policy_net(next_states).detach().max(1)[1].unsqueeze(dim=1).unsqueeze(dim=1).expand(
                self.batch_size, 1, self.n_atoms)
            ## 用 target 网络根据下一个状态计算的目标价值分布
            next_dist = self.target_net.dist(next_states).detach()
            ## 根据 action 选出对应最佳动作的采样概率 p_j
            next_dist = next_dist.gather(1, next_action).squeeze(dim=1)

            # calculate the distribution of the current state
            ## 计算真实的目标价值分布
            Tz = rewards + (1 - dones) * self.gamma * self.target_net.supports
            ## 将在价值范围外的值强制投射在 Vmin 和 Vmax 之间，即大于 Vmax 的等于 Vmax，小于 Vmin 的等于 Vmin,  T_{z_j}
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)
            ## 计算投影回原采样点后对应的原索引 b_j
            b = (Tz - self.Vmin) / self.policy_net.delta_z
            ## 记录该索引所在直方图左界的价值取值
            l = b.floor().long()
            ## 记录该索引所在直方图右界的价值取值
            u = b.ceil().long()
            ## 计算价值采样点的实际偏移量
            offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size).unsqueeze(dim=1).expand(
                self.batch_size, self.n_atoms).to(self.device)
            ## 根据 b 离 l 和 u 的距离进行加权，计算基于原始采样点上的目标概率
            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(0, torch.tensor(l + offset, dtype=torch.int).view(-1),
                                          (next_dist * (u.float() - b)).view(-1))
            proj_dist.view(-1).index_add_(0, torch.tensor(u + offset, dtype=torch.int).view(-1),
                                          (next_dist * (b - l.float())).view(-1))
        # calculate the loss
        ## 根据当前状态输出价值分布
        dist = self.policy_net.dist(states)
        actions = actions.unsqueeze(dim=1).expand(self.batch_size, 1, self.n_atoms)
        ## 根据 action 选出对应动作的采样概率
        dist = dist.gather(1, actions).squeeze(dim=1)
        # 计算 KL 散度
        loss = -(proj_dist * dist.log()).sum(1).mean()

        ## update the weight in the PER DQN
        ## 计算价值预估的差值，并根据该差值的大小，去更新样本的采样权重，差值越大，越容易被采样
        q_value_batch = torch.sum(proj_dist * self.target_net.supports, dim=1).unsqueeze(dim=1)
        expected_q_value_batch = torch.sum(dist * self.target_net.supports, dim=1).unsqueeze(dim=1)
        abs_errors = np.sum(
            np.abs(q_value_batch.cpu().detach().numpy() - expected_q_value_batch.cpu().detach().numpy()), axis=1)
        self.memory.batch_update(idxs_batch, abs_errors)

        # update the network
        ## 更新网络参数
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # soft update the target network
        ## 根据设置的 target_update 值，进行 target 网络和 policy 网络的参数同步
        if self.sample_count % self.target_update == 0:
            if self.tau == 1.0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            else:
                ## 通过软更新的方法，缓慢更新 target 网络的参数
                for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        ## 重置噪音
        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def save_model(self, fpath):
        '''
        保存模型
        Args:
            fpath (str): 模型存放路径
        '''
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.target_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        '''
        根据模型路径导入模型
        Args:
            fpath (str): 模型路径
        '''
        self.target_net.load_state_dict(torch.load(f"{fpath}/checkpoint.pt"))
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)
