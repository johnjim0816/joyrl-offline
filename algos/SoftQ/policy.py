import torch
import torch.nn as nn
import torch.nn.functional as F
import math,random
import numpy as np
from torch.distributions import Categorical
from algos.base.policies import BasePolicy
from algos.base.networks import QNetwork

class Policy(BasePolicy):
    def __init__(self,cfg) -> None:
        super(Policy, self).__init__(cfg)
        self.cfg = cfg
        self.alpha = cfg.alpha
        self.gamma = cfg.gamma  
        # e-greedy parameters
        self.sample_count = None
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.target_update = cfg.target_update
        self.create_graph() # create graph and optimizer
        self.create_summary() # create summary
        self.to(self.device)
        
    def create_graph(self):
        self.state_size, self.action_size = self.get_state_action_size()
        self.policy_net = QNetwork(self.cfg, self.state_size, self.action_size).to(self.device)
        self.target_net = QNetwork(self.cfg, self.state_size, self.action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict()) # or use this to copy parameters
        self.create_optimizer()

    def sample_action(self, state, **kwargs):
        ''' sample action
        '''
        # epsilon must decay(linear,exponential and etc.) for balancing exploration and exploitation
        self.sample_count = kwargs.get('sample_count')
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
            math.exp(-1. * self.sample_count / self.epsilon_decay) 
        if random.random() > self.epsilon:
            state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            q = self.policy_net(state)
            v = self.alpha * torch.log(torch.sum(torch.exp(q/self.alpha), dim=1, keepdim=True)).squeeze()
            dist = torch.exp((q-v)/self.alpha)
            dist = dist / torch.sum(dist)
            c = Categorical(dist)
            action = c.sample()
            action = action.item()
        else:
            action = self.action_space.sample()
        return action
    
    def predict_action(self, state, **kwargs):
        ''' predict action
        '''
        with torch.no_grad():
            state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            q = self.policy_net(state)
            v = self.alpha * torch.log(torch.sum(torch.exp(q/self.alpha), dim=1, keepdim=True)).squeeze()
            dist = torch.exp((q-v)/self.alpha)
            dist = dist / torch.sum(dist)
            action = torch.argmax(dist)
            action = action.item()
        return action
    
    def learn(self, **kwargs):
        ''' learn policy
        '''
        states, actions, next_states, rewards, dones = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones')
        update_step = kwargs.get('update_step')
        # convert numpy to tensor
        states = torch.tensor(states, device=self.device, dtype=torch.float32)
        actions = torch.tensor(actions, device=self.device, dtype=torch.int64).unsqueeze(dim=1)
        next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32)
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        with torch.no_grad():
            next_q = self.target_net(next_states)
            next_v = self.alpha * torch.log(torch.sum(torch.exp(next_q/self.alpha), dim=1, keepdim=True))
            y = rewards + (1 - dones) * self.gamma * next_v
        self.loss = F.mse_loss(self.policy_net(states).gather(1, actions.long()), y)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        # update target net every C steps
        if update_step % self.target_update == 0: 
            self.target_net.load_state_dict(self.policy_net.state_dict())
        self.update_summary() # update summary
 