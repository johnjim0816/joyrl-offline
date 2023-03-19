import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical,Normal
import torch.utils.data as Data
import numpy as np
import pickle

from common.models import ActorSoftmax, ActorNormal, Critic
from common.memories import PGReplay

class Agent:
    def __init__(self,cfg) -> None:
        self.device = torch.device(cfg.device)
        self.continuous = cfg.continuous # 连续动作空间
        self.action_space = cfg.action_space
        if self.continuous:
            self.action_scale = torch.tensor((self.action_space.high - self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            self.action_bias = torch.tensor((self.action_space.high + self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            self.policynet = ActorNormal(cfg.n_states,cfg.n_actions, hidden_dim = cfg.actor_hidden_dim).to(self.device)
        else:
            self.policynet = ActorSoftmax(cfg.n_states,cfg.n_actions, hidden_dim = cfg.actor_hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.policynet.parameters(), lr=cfg.lr)

        self.expert_path = f"algos/BC/{cfg.expert_path}"
        with open(self.expert_path, 'rb') as f:
            expert_data = pickle.load(f)
        self.expert_states = np.array(expert_data['states']) ; self.expert_actions = np.array(expert_data['actions'])


    @torch.no_grad()
    def predict_action(self,state):
        if self.continuous:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            mu, sigma = self.policynet(state)
            mean = mu * self.action_scale + self.action_bias
            std = sigma
            dist = Normal(mean, std)
            action = dist.sample()
            self.log_probs = dist.log_prob(action).detach()
            return action.detach().cpu().numpy()[0]
        else: 
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            probs = self.policynet(state)
            dist = Categorical(probs)
            action = dist.sample()
            self.log_probs = dist.log_prob(action).detach()
            return action.detach().cpu().numpy().item()
        
    def update(self, expert_states, expert_actions):
        expert_states = torch.tensor(expert_states, dtype=torch.float).to(self.device)
        expert_actions = torch.tensor(expert_actions).view(-1, 1).to(self.device)
        # 使用最大似然估计
        probs = self.policynet(expert_states).gather(1, expert_actions)
        log_probs = torch.log(probs)
        actor_loss = torch.mean(-log_probs)  
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def save_model(self, fpath):
        from pathlib import Path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.policynet.state_dict(), f"{fpath}/model.pth")

    def load_model(self, fpath):
        actor_ckpt = torch.load(f"{fpath}/model.pth", map_location=self.device)
        self.policynet.load_state_dict(actor_ckpt)
