import torch
import torch.nn as nn
import torch.nn.functional as F
import math, random
import numpy as np
from common.memories import ReplayBufferQue, ReplayBuffer
class DistributionalNetwork(nn.Module):
    def __init__(self, n_states, n_actions,n_atoms, Vmin, Vmax):
        '''
        值分布网络
        Args:
            n_states (int): 输入状态的维度
            n_action (int): 可执行动作的数量
            n_atoms (int): 用来描述分布的等间距的 atoms 的集合的大小 
            Vmin (float): atoms 集合中的最小值
            Vmax (float): atoms 集合中的最大值
        '''
        super(DistributionalNetwork, self).__init__()
        self.n_atoms = n_atoms  # number of atoms
        '''Vmin,Vmax: Range of the support of rewards. Ideally, it should be [min, max], '
                             'where min and max are referred to the min/max cumulative discounted '
                             'reward obtainable in one episode. Defaults to [0, 200].'
        '''
        ## support的取值范围
        self.Vmin = Vmin # minimum value of support
        self.Vmax = Vmax # maximum value of support
        self.delta_z = (Vmax - Vmin) / (n_atoms - 1) # 每个atom之间的间隔大小
        self.n_actions = n_actions

        self.fc1 = nn.Linear(n_states, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, n_actions * n_atoms)
        self.register_buffer('supports', torch.arange(Vmin, Vmax + self.delta_z, self.delta_z))
        # self.reset_parameters()
    def dist(self, x):
        '''
        计算 atoms 的分布
        Args:
            x (array): 状态
        Returns:
            x (array): 每个 atom 所对应的概率（即 support 的概率分布）
        '''
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.view(-1, self.n_actions, self.n_atoms)
        x = torch.softmax(x, dim=-1)
        return x

    def forward(self, x):
        x = self.dist(x)
        ## 计算 Q(x, a), 计算supports的期望
        x = torch.sum(x * self.supports, dim=2)
        return x
class Agent:
    def __init__(self,cfg) -> None:
        '''
        构建智能体
        Args:
            cfg (class): 超参数类 AlgoConfig
        '''
        self.n_actions = cfg.n_actions
        self.n_atoms = cfg.n_atoms
        self.Vmin = cfg.Vmin
        self.Vmax = cfg.Vmax
        self.gamma = cfg.gamma

        self.tau = cfg.tau
        self.device = torch.device(cfg.device)

        self.policy_net = DistributionalNetwork(cfg.n_states, cfg.n_actions, cfg.n_atoms, cfg.Vmin, cfg.Vmax).to(self.device) # 策略网络
        self.target_net= DistributionalNetwork(cfg.n_states, cfg.n_actions, cfg.n_atoms, cfg.Vmin, cfg.Vmax).to(self.device) # 目标网络，在训练过程中软更新
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.buffer_size) # ReplayBufferQue(cfg.capacity)
        self.sample_count = 0

        ## epsilon相关参数，探索与利用平衡
        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update # 目标网络更新频率

    def sample_action(self, state):
        ''' 
        采样动作
        Args:
            state (array): 状态
        Returns:
            action (int): 动作
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
        预测动作
        Args:
            state (array): 状态
        Returns:
            action (int): 动作
        '''
        with torch.no_grad():
            state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            # print ("state", state)
            q_values = self.policy_net(state)
            action  = q_values.max(1)[1].item()
            # action = q_values.argmax() // self.n_atoms
            # action = action.item()  # choose action corresponding to the maximum q value
        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(dim=1)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        # calculate the distribution of the next state
        
        with torch.no_grad():
            ## 计算每个 batch 下一步的动作与该动作对应的 atoms 分布
            next_action = self.policy_net(next_states).detach().max(1)[1].unsqueeze(dim=1).unsqueeze(dim=1).expand(self.batch_size, 1, self.n_atoms)
            # next_dist.shape=(batch_size, n_actions, n_atoms)
            next_dist = self.target_net.dist(next_states).detach()
            # next_dist.shape=(batch_size, n_atoms)
            next_dist = next_dist.gather(1, next_action).squeeze(dim=1)

            # calculate the distribution of the current state
            ## 贝尔曼更新的公式为: Tz = r + gamma * z
            Tz = rewards + (1 - dones) * self.gamma * self.target_net.supports
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax) # 将 Tz 值保持在 V_max 和 V_min 之间
            b = (Tz - self.Vmin) / self.policy_net.delta_z # b 位于 [0, N-1] 之间
            l = b.floor().long()
            u = b.ceil().long()
            ## 计算 Phi Tz，即投影贝尔曼更新
            # 后续计算 proj_dist 时, 将其拉平为一维向量, 需要重新计算每个 batch 索引的起始点
            offset = torch.linspace(0, (self.batch_size - 1) * self.n_atoms, self.batch_size).unsqueeze(dim=1).expand(self.batch_size, self.n_atoms).to(self.device)
            proj_dist = torch.zeros(next_dist.size(), device=self.device) # 初始化投影概率分布
            ## 将 next_dist 按照一定投影规则（参见下两行代码）分配到atom上
            proj_dist.view(-1).index_add_(0, torch.tensor(l + offset,dtype=torch.int).view(-1), (next_dist * (u.float() - b)).view(-1)) # 该比例为 u-b
            proj_dist.view(-1).index_add_(0, torch.tensor(u + offset,dtype=torch.int).view(-1), (next_dist * (b - l.float())).view(-1)) # 该比例为 b-l
        ## 计算 current state 下的 atoms 的概率分布
        dist = self.policy_net.dist(states)
        actions = actions.unsqueeze(dim=1).expand(self.batch_size, 1, self.n_atoms)
        dist = dist.gather(1, actions).squeeze(dim=1)
        ## cross-entropy 损失
        loss = -(proj_dist * dist.log()).sum(1).mean()
        # update the network
        self.optimizer.zero_grad()
        loss.backward()
        ## 防止梯度爆炸而对梯度进行的裁剪，类似 torch.clamp() 功能
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        # soft update the target network
        if self.sample_count % self.target_update == 0:
            if self.tau == 1.0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
            else:
                for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
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


