#!/usr/bin/env python
# coding=utf-8
'''
Author: JiangJi
Email: johnjim0816@gmail.com
Date: 2023-02-21 20:32:11
LastEditor: Scc_hy
LastEditTime: 2023-05-28 11:26:11
Discription: 
'''
import math,random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from algos.base.policies import BasePolicy
from algos.base.networks import CriticNetwork, ActorNetwork
from algos.base.buffers import ReplayBufferQue
from algos.base.optms import SharedAdam


class Policy(BasePolicy):
    def __init__(self, cfg, is_share_agent=False):
        super(Policy, self).__init__(cfg)
        self.cfg = cfg
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
        self.action_scale = torch.tensor((self.action_space.high - self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        self.action_bias = torch.tensor((self.action_space.high + self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        self.create_graph() # create graph and optimizer
        self.create_summary() # create summary

        if is_share_agent:
            self.actor.share_memory()
            self.actor_optimizer = SharedAdam(self.actor.parameters(), lr=cfg.actor_lr)
            self.actor_optimizer.share_memory()
            self.critic_1.share_memory()
            self.critic_1_optimizer = SharedAdam(self.critic_1.parameters(), lr=cfg.critic_lr)
            self.critic_1_optimizer.share_memory()
            self.critic_2.share_memory()
            self.critic_2_optimizer = SharedAdam(self.critic_2.parameters(), lr=cfg.critic_lr)
            self.critic_2_optimizer.share_memory()
    
    def create_graph(self):
        # Actor
        self.actor = ActorNetwork(self.cfg, self.actor_input_dim, self.actor_output_dim).to(self.device)
        self.actor_target = ActorNetwork(self.cfg, self.actor_input_dim, self.actor_output_dim).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        # critice - 2Q
        self.critic_1 = CriticNetwork(self.cfg, self.critic_input_dim).to(self.device)
        self.critic_2 = CriticNetwork(self.cfg, self.critic_input_dim).to(self.device)
        self.critic_1_target = CriticNetwork(self.cfg, self.critic_input_dim).to(self.device)
        self.critic_2_target = CriticNetwork(self.cfg, self.critic_input_dim).to(self.device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.create_optimizer() 

    def create_optimizer(self):
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr = self.actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr = self.critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr = self.critic_lr)

    def create_summary(self):
        '''
        创建 tensorboard 数据
        '''
        self.summary = {
            'scalar': {
                'tot_loss': 0.0,
                'policy_loss': 0.0,
                'value_loss1': 0.0,
                'value_loss2': 0.0,
            },
        }

    def update_summary(self):
        ''' 更新 tensorboard 数据
        '''
        if hasattr(self, 'tot_loss'):
            self.summary['scalar']['tot_loss'] = self.tot_loss.item()
        try:
            self.summary['scalar']['policy_loss'] = self.policy_loss.item()
        except Exception as e:
            pass
        self.summary['scalar']['value_loss1'] = self.value_loss1.item()
        self.summary['scalar']['value_loss2'] = self.value_loss2.item()

    def sample_action(self, state,  **kwargs):
        self.sample_count += 1
        if self.sample_count < self.explore_steps:
            return self.action_space.sample()
        else:
            state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            # action = torch.tanh(self.actor(state))
            action = self.actor(state)
            action = self.action_scale * action + self.action_bias
            action = action.detach().cpu().numpy()[0]
            action_noise = np.random.normal(0, self.action_scale.cpu().numpy()[0] * self.expl_noise, size=self.n_actions)
            action = (action + action_noise).clip(self.action_space.low, self.action_space.high)
            return action

    @torch.no_grad()
    def predict_action(self, state,  **kwargs):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        action = self.actor(state)
        action = self.action_scale * action + self.action_bias
        return action.detach().cpu().numpy()[0]

    def train(self, share_agent=None, **kwargs):
        # if len(self.memory) < self.explore_steps:
        #     return
        if kwargs.get('update_step') < self.explore_steps:
            return 
        # state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        state, action, next_state, reward, done = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones')
        
        # convert to tensor
        state = torch.tensor(np.array(state), device=self.device, dtype=torch.float32)
        action = torch.tensor(np.array(action), device=self.device, dtype=torch.float32)
        next_state = torch.tensor(np.array(next_state), device=self.device, dtype=torch.float32)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32).unsqueeze(1)
        done = torch.tensor(done, device=self.device, dtype=torch.float32).unsqueeze(1)
        # update critic
        noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        # print ("")
        next_action = ((self.actor_target(next_state) + noise) * self.action_scale + self.action_bias).clamp(-self.action_scale+self.action_bias, self.action_scale+self.action_bias)
        next_sa = torch.cat([next_state, next_action], 1) # shape:[train_batch_size,n_states+n_actions]
        target_q1, target_q2 = self.critic_1_target(next_sa).detach(), self.critic_2_target(next_sa).detach()
        target_q = torch.min(target_q1, target_q2) # shape:[train_batch_size,n_actions]
        target_q = reward + self.gamma * target_q * (1 - done)
        sa = torch.cat([state, action], 1)
        current_q1, current_q2 = self.critic_1(sa), self.critic_2(sa)
        # compute critic loss
        critic_1_loss = F.mse_loss(current_q1, target_q)
        critic_2_loss = F.mse_loss(current_q2, target_q)
        self.value_loss1, self.value_loss2 = critic_1_loss, critic_2_loss

        if share_agent is not None:
            share_agent.critic_1_optimizer.zero_grad()
            share_agent.critic_2_optimizer.zero_grad()
            
            self.critic_1_optimizer.zero_grad()  		
            self.critic_2_optimizer.zero_grad()  

            critic_1_loss.backward()
            critic_2_loss.backward()

            for param, share_param in zip(self.critic_1.parameters(), share_agent.critic_1.parameters()):
                share_param._grad = param.grad   			
            for param, share_param in zip(self.critic_2.parameters(), share_agent.critic_2.parameters()):
                share_param._grad = param.grad 

            share_agent.critic_1_optimizer.step()   	
            share_agent.critic_2_optimizer.step()   

            self.critic_1.load_state_dict(share_agent.critic_1.state_dict())
            self.critic_2.load_state_dict(share_agent.critic_2.state_dict())    

            # if self.sample_count % self.policy_freq == 0 or self.sample_count % self.policy_freq == 1:
                # print ("self.sample_count = ", self.sample_count)
            # compute actor loss
            share_agent.actor_optimizer.zero_grad()
            actor_loss = -self.critic_1(torch.cat([state, self.actor(state)], 1)).mean()
            self.policy_loss = actor_loss
            self.tot_loss = self.policy_loss + self.value_loss1 + self.value_loss2
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            for param, share_param in zip(self.actor.parameters(), share_agent.actor.parameters()):
                share_param._grad = param.grad   
            share_agent.actor_optimizer.step()     
            self.actor.load_state_dict(share_agent.actor.state_dict())

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)				  
        else :
            self.critic_1_optimizer.zero_grad()
            critic_1_loss.backward()
            self.critic_1_optimizer.step()
            self.critic_2_optimizer.zero_grad()
            critic_2_loss.backward()
            self.critic_2_optimizer.step()
            # Delayed policy updates
            if self.sample_count % self.policy_freq == 0:
                # compute actor loss
                actor_loss = -self.critic_1(torch.cat([state, self.actor(state)], 1)).mean()
                self.policy_loss = actor_loss
                self.tot_loss = self.policy_loss + self.value_loss1 + self.value_loss2
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        self.update_summary()

    def save_model(self, fpath):
        from pathlib import Path
        # create path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.critic_1.state_dict(), f"{fpath}/critic_1.pth")
        torch.save(self.critic_2.state_dict(), f"{fpath}/critic_2.pth")
        torch.save(self.actor.state_dict(), f"{fpath}/actor.pth")

    def load_model(self, fpath):
        critic_1_ckpt = torch.load(f"{fpath}/critic_1.pth", map_location=self.device)
        critic_2_ckpt = torch.load(f"{fpath}/critic_2.pth", map_location=self.device)
        actor_ckpt = torch.load(f"{fpath}/actor.pth", map_location=self.device)
        self.critic_1.load_state_dict(critic_1_ckpt)
        self.critic_2.load_state_dict(critic_2_ckpt)
        self.actor.load_state_dict(actor_ckpt)
        
