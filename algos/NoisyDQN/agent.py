import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import random
import ray
from common.memories import ReplayBuffer
import torch
import torch.nn as nn
import torch.optim as optim
from common.optms import SharedAdam


class NoisyLinear(nn.Module):
    '''在Noisy DQN中用NoisyLinear层替换普通的nn.Linear层
    '''
    def __init__(self, in_dim, out_dim, std_init=0.4):
        super(NoisyLinear, self).__init__()
        
        self.in_dim  = in_dim
        self.out_dim = out_dim
        self.std_init  = std_init
        
        self.weight_mu    = nn.Parameter(torch.empty(out_dim, in_dim))
        self.weight_sigma = nn.Parameter(torch.empty(out_dim, in_dim))
        # 将一个 tensor 注册成 buffer，使得这个 tensor 不被当做模型参数进行优化。
        self.register_buffer('weight_epsilon', torch.empty(out_dim, in_dim)) 
        
        self.bias_mu    = nn.Parameter(torch.empty(out_dim))
        self.bias_sigma = nn.Parameter(torch.empty(out_dim))
        self.register_buffer('bias_epsilon', torch.empty(out_dim))
        
        self.reset_parameters() # 初始化参数
        self.reset_noise()  # 重置噪声
    
    def forward(self, x):
        if self.training: 
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)
    
    def reset_parameters(self):
        mu_range = 1 / self.in_dim ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / self.in_dim ** 0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / self.out_dim ** 0.5)
    
    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_dim)
        epsilon_out = self._scale_noise(self.out_dim)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_dim))
    
    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class NoisyQNetwork(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim=128):
        super(NoisyQNetwork, self).__init__()
        self.fc1 =  nn.Linear(n_states, hidden_dim)
        self.noisy_fc2 = NoisyLinear(hidden_dim, hidden_dim)
        self.noisy_fc3 = NoisyLinear(hidden_dim, n_actions)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.noisy_fc2(x))
        x = self.noisy_fc3(x)
        return x

    def reset_noise(self):
        self.noisy_fc2.reset_noise()
        self.noisy_fc3.reset_noise()

class Agent:
    def __init__(self, cfg, is_share_agent = False) -> None:
        '''智能体类
        Args:
            cfg (class): 超参数类
            is_share_agent (bool, optional): 是否为共享的 Agent ，多进程下使用，默认为 False
        '''
        self.n_actions = cfg.n_actions  
        self.device = torch.device(cfg.device) 
        self.gamma = cfg.gamma  
        ## e-greedy parameters
        self.sample_count = 0  # sample count for epsilon decay
        self.epsilon = cfg.epsilon_start
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update
        self.device = torch.device(cfg.device) 

        self.policy_net = NoisyQNetwork(cfg.n_states,cfg.n_actions,hidden_dim=cfg.hidden_dim).to(self.device)
        self.target_net = NoisyQNetwork(cfg.n_states,cfg.n_actions,hidden_dim=cfg.hidden_dim).to(self.device)

        # 设置模型的训练和测试模式，主要是对于noisylinear，或者一些dropout和batch normalization层
        if cfg.mode == 'train':
            self.policy_net.train()
            self.target_net.train()
        elif cfg.mode == 'test':
            self.policy_net.eval()
            self.target_net.eval()

        # copy parameters from policy net to target net
        for target_param, param in zip(self.target_net.parameters(),self.policy_net.parameters()): 
            target_param.data.copy_(param.data)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr)
        self.memory = ReplayBuffer(cfg.buffer_size)
        if is_share_agent:
            self.policy_net.share_memory()
            self.optimizer = SharedAdam(self.policy_net.parameters(), lr=cfg.lr)
            self.optimizer.share_memory()
        # ray中share_agent
        self.share_policy_ray = NoisyQNetwork(cfg.n_states,cfg.n_actions,hidden_dim=cfg.hidden_dim).to(self.device)
        self.share_target_ray = NoisyQNetwork(cfg.n_states,cfg.n_actions,hidden_dim=cfg.hidden_dim).to(self.device)
        self.optimizer_ray = SharedAdam(self.share_policy_ray.parameters(), lr=cfg.lr)

    def sample_action(self, state):
        ''' sample action with e-greedy policy 
        '''
        self.sample_count += 1
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            with torch.no_grad():
                state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
                q_values = self.policy_net(state)
                action = q_values.max(1)[1].item() # choose action corresponding to the maximum q value
        else:
            action = random.randrange(self.n_actions)
        return action

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        q_value = self.policy_net(state)
        action  = q_value.max(1)[1].item()
        return action
    
    def update(self, share_agent=None):
        if len(self.memory) < self.batch_size: # when transitions in memory donot meet a batch, not update
            return
        # beta = min(1.0, self.beta_start + self.sample_count * (1.0 - self.beta_start) / self.beta_frames)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        # state_batch, action_batch, reward_batch, next_state_batch, done_batch, weights_batch, indices = self.memory.sample(self.batch_size, beta) 
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float) 
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        done_batch = torch.tensor(done_batch, device=self.device, dtype=torch.float).unsqueeze(1)
        # weights_batch = torch.tensor(weights_batch, device=self.device, dtype=torch.float)
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch) # shape(batchsize,1),requires_grad=True
        next_max_q_value_batch = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1) 
        expected_q_value_batch = reward_batch + self.gamma * next_max_q_value_batch* (1-done_batch)
        
        loss = nn.MSELoss()(q_value_batch, expected_q_value_batch)  # shape same to  
        # backpropagation
        if share_agent is not None:
            # Clear the gradient of the previous step of share_agent
            share_agent.optimizer.zero_grad()
            loss.backward()
            # clip to avoid gradient explosion
            for param in self.policy_net.parameters():  
                param.grad.data.clamp_(-1, 1)
            # Copy the gradient from policy_net of local_agnet to policy_net of share_agent
            for param, share_param in zip(self.policy_net.parameters(), share_agent.policy_net.parameters()):
                share_param._grad = param.grad
            share_agent.optimizer.step()
            self.policy_net.load_state_dict(share_agent.policy_net.state_dict())
            if self.sample_count % self.target_update == 0: # target net update, target_update means "C" in pseucodes
                self.target_net.load_state_dict(self.policy_net.state_dict())
        else:
            self.optimizer.zero_grad()  
            loss.backward()
            # clip to avoid gradient explosion
            for param in self.policy_net.parameters():  
                param.grad.data.clamp_(-1, 1)
            self.optimizer.step() 

            if self.sample_count % self.target_update == 0: # target net update, target_update means "C" in pseucodes
                self.target_net.load_state_dict(self.policy_net.state_dict())   

        self.policy_net.reset_noise()
        self.target_net.reset_noise()

    def update_ray(self, share_policy_state_dict):
        """Update the share_agent parameters with ray"""
        if len(self.memory) < self.batch_size: # when transitions in memory donot meet a batch, not update
            return share_policy_state_dict
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.memory.sample(
            self.batch_size)
        state_batch = torch.tensor(np.array(state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        action_batch = torch.tensor(action_batch, device=self.device).unsqueeze(1) # shape(batchsize,1)
        reward_batch = torch.tensor(reward_batch, device=self.device, dtype=torch.float).unsqueeze(1) # shape(batchsize,1)
        next_state_batch = torch.tensor(np.array(next_state_batch), device=self.device, dtype=torch.float) # shape(batchsize,n_states)
        done_batch = torch.tensor(np.float32(done_batch), device=self.device).unsqueeze(1) # shape(batchsize,1)
        # print(state_batch.shape,action_batch.shape,reward_batch.shape,next_state_batch.shape,done_batch.shape)
        # compute current Q(s_t,a), it is 'y_j' in pseucodes
        q_value_batch = self.policy_net(state_batch).gather(dim=1, index=action_batch) # shape(batchsize,1),requires_grad=True
        # print(q_values.requires_grad)
        # compute max(Q(s_t+1,A_t+1)) respects to actions A, next_max_q_value comes from another net and is just regarded as constant for q update formula below, thus should detach to requires_grad=False
        next_max_q_value_batch = self.target_net(next_state_batch).max(1)[0].detach().unsqueeze(1) 
        # print(q_values.shape,next_q_values.shape)
        # compute expected q value, for terminal state, done_batch[0]=1, and expected_q_value=rewardcorrespondingly
        expected_q_value_batch = reward_batch + self.gamma * next_max_q_value_batch* (1-done_batch)
        # print(expected_q_value_batch.shape,expected_q_value_batch.requires_grad)
        loss = nn.MSELoss()(q_value_batch, expected_q_value_batch)
        # 加载最新的共享智能体的网络参数
        self.share_policy_ray.load_state_dict(share_policy_state_dict)
        self.optimizer_ray.zero_grad()
        loss.backward()
        # clip to avoid gradient explosion
        for param in self.policy_net.parameters():  
            param.grad.data.clamp_(-1, 1)
        # 将local_agent计算出的梯度参数传给share_policy_ray
        for param, share_param in zip(self.policy_net.parameters(), self.share_policy_ray.parameters()):
            share_param._grad = param.grad
        # 更新share_policy_ray网络的参数
        self.optimizer_ray.step()
        # 将更新后的share_policy_ray网络的参数传给local_agent
        self.policy_net.load_state_dict(self.share_policy_ray.state_dict())
        if self.sample_count % self.target_update == 0: # target net update, target_update means "C" in pseucodes
            self.target_net.load_state_dict(self.policy_net.state_dict())

        self.policy_net.reset_noise()
        self.target_net.reset_noise()
        
        # 将更新后的share_policy_ray网络的参数传回ShareAgent类
        return self.share_policy_ray.state_dict()
    
    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        checkpoint = torch.load(f"{fpath}/checkpoint.pt", map_location=self.device)
        self.target_net.load_state_dict(checkpoint)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)


@ray.remote
class ShareAgent:
    def __init__(self, cfg):
        '''共享智能体类
        Args:
            cfg (class): 超参数类
        '''
        self.policy_net = NoisyQNetwork(cfg.n_states,cfg.n_actions,hidden_dim=cfg.hidden_dim).to(cfg.device)
        self.target_net = NoisyQNetwork(cfg.n_states,cfg.n_actions,hidden_dim=cfg.hidden_dim).to(cfg.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) 
        self.lr = cfg.lr
        # self.memory = ReplayBuffer(cfg.buffer_size)

    def get_parameters(self):
        return self.policy_net.state_dict()
    
    def receive_parameters(self, policy_net):
        self.policy_net.load_state_dict(policy_net)

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.policy_net.state_dict(), f"{fpath}/checkpoint.pt")

    def load_model(self, fpath):
        checkpoint = torch.load(f"{fpath}/checkpoint.pt")
        self.target_net.load_state_dict(checkpoint)
        for target_param, param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            param.data.copy_(target_param.data)

    def update_parameters(self, local_net):
        """training algorithm in ShareAgent"""
        self.optimizer.zero_grad()
        for param, share_param in zip(local_net.parameters(), self.policy_net.parameters()):
            share_param._grad = param.grad
        self.optimizer.step()
        return self.policy_net