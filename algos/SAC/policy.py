import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical,Normal
import torch.utils.data as Data
import numpy as np

from algos.base.networks import ValueNetwork, CriticNetwork, ActorNetwork
from algos.base.policies import BasePolicy

class Policy(BasePolicy):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.independ_actor = cfg.independ_actor
        self.share_optimizer = cfg.share_optimizer
        self.gamma = cfg.gamma
        self.action_type = cfg.action_type
        if self.action_type.lower() == 'continuous': # continuous action space
            self.action_scale = torch.tensor((self.action_space.high - self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            self.action_bias = torch.tensor((self.action_space.high + self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.batch_size = cfg.batch_size
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.target_entropy = -torch.prod(torch.Tensor(self.action_space.shape).to(self.device)).item()
        self.create_graph()
        self.create_optimizer()
        self.create_summary()
        self.to(self.device)

    def create_summary(self):
        '''
        创建 tensorboard 数据
        '''
        self.summary = {
            'scalar': {
                'tot_loss': 0.0,
                'actor_loss': 0.0,
                'critic1_loss': 0.0,
                'critic2_loss': 0.0,
                'alpha_loss': 0.0,
            },
        }

    def update_summary(self):
        ''' 更新 tensorboard 数据
        '''
        if hasattr(self, 'tot_loss'):
            self.summary['scalar']['tot_loss'] = self.tot_loss.item()
        self.summary['scalar']['actor_loss'] = self.actor_loss.item()
        self.summary['scalar']['critic1_loss'] = self.critic1_loss.item()
        self.summary['scalar']['critic2_loss'] = self.critic2_loss.item()
        self.summary['scalar']['alpha_loss'] = self.alpha_loss.item()

    def create_graph(self):
        self.state_size, self.action_size = self.get_state_action_size()
        if not self.independ_actor:
            self.policy_net = ValueNetwork(self.cfg, self.state_size, self.action_space)
        else:
            self.actor = ActorNetwork(self.cfg, self.state_size, self.action_space)
            if self.action_type.lower() == 'continuous':
                self.input_head_size = [None, self.state_size[-1]+self.action_size[-1]]
                self.critic1 = CriticNetwork(self.cfg, self.input_head_size).to(self.device)
                self.critic2 = CriticNetwork(self.cfg, self.input_head_size).to(self.device) 
                self.target_critic1 = CriticNetwork(self.cfg, self.input_head_size).to(self.device)
                self.target_critic2 = CriticNetwork(self.cfg, self.input_head_size).to(self.device)
            else:
                self.critic1 = CriticNetwork(self.cfg, self.state_size, self.action_size[-1]).to(self.device)
                self.critic2 = CriticNetwork(self.cfg, self.state_size, self.action_size[-1]).to(self.device)
                self.target_critic1 = CriticNetwork(self.cfg, self.state_size, self.action_size[-1]).to(self.device)
                self.target_critic2 = CriticNetwork(self.cfg, self.state_size, self.action_size[-1]).to(self.device)
            self.target_critic1.load_state_dict(self.critic1.state_dict())
            self.target_critic2.load_state_dict(self.critic2.state_dict())
            
    def create_optimizer(self):
        if self.share_optimizer:
            self.optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr)
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
            self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.cfg.critic1_lr)
            self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.cfg.critic2_lr)
            self.log_alpha_optimizer = optim.Adam([self.log_alpha], lr=self.cfg.alpha_lr)

    def get_action(self, state, mode='sample', **kwargs):
        state = np.array(state)
        if len(state.shape) == 1: state = state[np.newaxis, :]
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        if not self.independ_actor:
            if self.action_type.lower() == 'continuous':
                self.value, self.mu, self.sigma = self.policy_net(state)
            else:
                self.probs = self.policy_net(state)
        else:
            # self.value = self.critic1(state)
            self.value = [None]
            output = self.actor(state) 
            if self.action_type.lower() == 'continuous':
                self.mu, self.sigma = output['mu'], output['sigma']
            else:
                self.probs = output['probs']
        if mode == 'sample':
            action = self.sample_action(**kwargs)
            self.update_policy_transition()
        elif mode == 'predict':
            action = self.predict_action(**kwargs)
        else:
            raise NameError('mode must be sample or predict')
        return action
    
    def update_policy_transition(self):
        if self.action_type.lower() == 'continuous':
            self.policy_transition = {'value': self.value, 'mu': self.mu, 'sigma': self.sigma}
        else:
            self.policy_transition = {'value': self.value, 'probs': self.probs, 'log_probs': self.log_probs}
    
    def sample_action(self,**kwargs):
        if self.action_type.lower() == 'continuous':
            action, self.log_probs = self.calc_log_prob(self.mu, self.sigma)
            return action.detach().cpu().numpy()[0]
        else:
            dist = Categorical(self.probs)
            action = dist.sample()
            self.log_probs = dist.log_prob(action)
            return action.detach().cpu().numpy()[0]
        
    def predict_action(self, **kwargs):
        if self.action_type.lower() == 'continuous':
            return self.mu.detach().cpu().numpy()[0]
        else:
            return torch.argmax(self.probs).detach().cpu().numpy()
        
    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)
            
    def calc_log_prob(self, mu, sigma):
        dist = Normal(mu, sigma)
        normal_sample = dist.rsample() 
        log_prob = dist.log_prob(normal_sample)
        action = torch.tanh(normal_sample)
        log_prob = log_prob - torch.log(1 - torch.tanh(action).pow(2) + 1e-7)
        action = action * self.action_space.high[0]
        return action, log_prob

    def calc_target(self, rewards, next_states, dones):
        if self.action_type.lower() == 'continuous': 
            output = self.actor(next_states)
            mu, sigma = output['mu'], output['sigma']
            next_actions, log_probs = self.calc_log_prob(mu, sigma)
            entropy = -log_probs
            q1_value = self.target_critic1(torch.cat([next_states, next_actions], 1).to(self.device))
            q2_value = self.target_critic2(torch.cat([next_states, next_actions], 1).to(self.device))
            next_value = torch.min(q1_value,
                                q2_value) + self.log_alpha.exp() * entropy
            td_target = rewards + self.gamma * next_value * (1 - dones)
            return td_target
        else:
            output = self.actor(next_states)
            next_probs = output['probs']
            next_log_probs = torch.log(next_probs + 1e-8)
            entropy = -torch.sum(next_probs * next_log_probs, dim=1, keepdim=True)
            q1_value = self.target_critic1(next_states)
            q2_value = self.target_critic2(next_states)
            min_qvalue = torch.sum(next_probs * torch.min(q1_value, q2_value),
                                dim=1,
                                keepdim=True)
            next_value = min_qvalue + self.log_alpha.exp() * entropy
            td_target = rewards + self.gamma * next_value * (1 - dones)
            return td_target

    def learn(self, **kwargs): 
        states, actions, next_states, rewards, dones = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones')
        # convert to tensor
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32) # shape:[batch_size,n_states]
        actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.int64) # shape:[batch_size,n_actions]
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float32) # shape:[batch_size,n_states]
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float32) # shape:[batch_size,1]
        dones = torch.tensor(np.array(dones), device=self.device, dtype=torch.float32) # shape:[batch_size,1]
        if self.action_type.lower() == 'continuous':   
            # update Q net
            td_target = self.calc_target(rewards, next_states, dones)
            self.critic1_loss = torch.mean(
                F.mse_loss(self.critic1(torch.cat([states, actions], 1).to(self.device)), td_target.detach()))
            self.critic2_loss = torch.mean(
                F.mse_loss(self.critic2(torch.cat([states, actions], 1).to(self.device)), td_target.detach()))
            self.critic1_optimizer.zero_grad()
            self.critic1_loss.backward()
            self.critic1_optimizer.step()
            self.critic2_optimizer.zero_grad()
            self.critic2_loss.backward()
            self.critic2_optimizer.step()
            # update policy net
            output = self.actor(states)
            mu, sigma = output['mu'], output['sigma']
            new_actions, log_probs = self.calc_log_prob(mu, sigma)
            log_probs = log_probs.detach()
            entropy = -log_probs
            q1_value = self.critic1(torch.cat([states, new_actions], 1).to(self.device))
            q2_value = self.critic2(torch.cat([states, new_actions], 1).to(self.device))
            self.actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                    torch.min(q1_value, q2_value))
            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_optimizer.step()
            # update alpha
            self.alpha_loss = torch.mean(
                (entropy - self.target_entropy).detach() * self.log_alpha.exp())
            self.log_alpha_optimizer.zero_grad()
            self.alpha_loss.backward()
            self.log_alpha_optimizer.step()
            self.soft_update(self.critic1, self.target_critic1)
            self.soft_update(self.critic2, self.target_critic2)
        else:
            # update Q net
            td_target = self.calc_target(rewards, next_states, dones)
            critic_1_q_values = self.critic1(states).gather(1, actions)
            self.critic1_loss = torch.mean(
                F.mse_loss(critic_1_q_values, td_target.detach()))
            critic_2_q_values = self.critic2(states).gather(1, actions)
            self.critic2_loss = torch.mean(
                F.mse_loss(critic_2_q_values, td_target.detach()))
            self.critic1_optimizer.zero_grad()
            self.critic1_loss.backward()
            self.critic1_optimizer.step()
            self.critic2_optimizer.zero_grad()
            self.critic2_loss.backward()
            self.critic2_optimizer.step()
            # update policy net
            output = self.actor(states)
            probs = output['probs']
            log_probs = torch.log(probs + 1e-8)
            entropy = -torch.sum(probs * log_probs, dim=1, keepdim=True)  #
            q1_value = self.critic1(states)
            q2_value = self.critic2(states)
            min_qvalue = torch.sum(probs * torch.min(q1_value, q2_value),
                                dim=1,
                                keepdim=True)
            self.actor_loss = torch.mean(-self.log_alpha.exp() * entropy - min_qvalue)
            self.actor_optimizer.zero_grad()
            self.actor_loss.backward()
            self.actor_optimizer.step()
            target_entropy = -1
            # update alpha
            self.alpha_loss = torch.mean(
                (entropy - target_entropy).detach() * self.log_alpha.exp())
            self.log_alpha_optimizer.zero_grad()
            self.alpha_loss.backward()
            self.log_alpha_optimizer.step()
            self.soft_update(self.critic1, self.target_critic1)
            self.soft_update(self.critic2, self.target_critic2)
        self.update_summary() # update summary

        