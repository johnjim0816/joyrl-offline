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
        self.gamma = cfg.gamma
        self.entropy_coef = cfg.entropy_coef
        self.independ_actor = cfg.independ_actor
        self.share_optimizer = cfg.share_optimizer
        self.action_type = cfg.action_type
        if self.action_type.lower() == 'continuous': # continuous action space
            self.action_scale = torch.tensor((self.action_space.high - self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
            self.action_bias = torch.tensor((self.action_space.high + self.action_space.low)/2, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        self.k_epochs = cfg.k_epochs # update policy for K epochs
        self.batch_size = cfg.batch_size
        self.sgd_batch_size = cfg.sgd_batch_size
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
                'critic_loss': 0.0,
            },
        }

    def update_summary(self):
        ''' 更新 tensorboard 数据
        '''
        if hasattr(self, 'tot_loss'):
            self.summary['scalar']['tot_loss'] = self.tot_loss.item()
        self.summary['scalar']['actor_loss'] = self.actor_loss.item()
        self.summary['scalar']['critic_loss'] = self.critic_loss.item()

    def create_graph(self):
        self.state_size, self.action_size = self.get_state_action_size()
        if not self.independ_actor:
            self.policy_net = ValueNetwork(self.cfg, self.state_size, self.action_space).to(self.device)
        else:
            self.actor = ActorNetwork(self.cfg, self.state_size, self.action_space).to(self.device)
            self.critic = CriticNetwork(self.cfg, self.state_size).to(self.device)

    def create_optimizer(self):
        if self.share_optimizer:
            self.optimizer = optim.Adam(self.parameters(), lr=self.cfg.lr)
        else:
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

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
            self.value = self.critic(state)
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
            mean = self.mu * self.action_scale + self.action_bias
            std = self.sigma
            dist = Normal(mean,std)
            action = dist.sample()
            action = torch.clamp(action, torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32), torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32))
            self.log_probs = dist.log_prob(action).detach()
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
        
    def evaluate(self, states, actions):
        if not self.independ_actor:
            if self.action_type.lower() == 'continuous':
                values, mu, sigma = self.policy_net(states)
            else:
                probs = self.policy_net(states)
        else:
            values = self.critic(states)
            output = self.actor(states)
        if self.action_type.lower() == 'continuous':
            mu , sigma = output['mu'], output['sigma']
            mean = mu * self.action_scale + self.action_bias
            std = sigma
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions)
        else:
            probs = output['probs']
            dist = Categorical(probs)
            log_probs = torch.log(probs.gather(1, actions))
        entropies = dist.entropy()
        return values, log_probs, entropies   

    def learn(self, **kwargs): 
        states, actions, next_states, rewards, dones = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones')
        if self.action_type.lower() == 'continuous':
            mus, sigmas = kwargs.get('mu'), kwargs.get('sigma')
            mus = torch.stack(mus, dim=0).to(device=self.device, dtype=torch.float32)
            sigmas = torch.stack(sigmas, dim=0).to(device=self.device, dtype=torch.float32)
            means = mus * self.action_scale + self.action_bias
            stds = sigmas
            dists = Normal(means,stds)
            old_log_probs = dists.log_prob(torch.tensor(np.array(actions), device=self.device, dtype=torch.float32)).detach()
            old_probs = torch.exp(old_log_probs)
        else:
            old_probs, old_log_probs  = kwargs.get('probs'), kwargs.get('log_probs')
            old_probs = torch.cat(old_probs,dim=0).to(self.device) # shape:[batch_size,n_actions]
            old_log_probs = torch.cat(old_log_probs,dim=0).to(self.device).unsqueeze(dim=1) # shape:[batch_size,1]
        # convert to tensor
        states = torch.tensor(np.array(states), device=self.device, dtype=torch.float32) # shape:[batch_size,n_states]
        if self.action_type.lower() == 'continuous':
            actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.float32) # shape:[batch_size,1]
        else:
            actions = torch.tensor(np.array(actions), device=self.device, dtype=torch.int64) # shape:[batch_size,1]
        next_states = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float32) # shape:[batch_size,n_states]
        rewards = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float32) # shape:[batch_size,1]
        dones = torch.tensor(np.array(dones), device=self.device, dtype=torch.float32) # shape:[batch_size,1]
        returns = self._compute_returns(rewards, dones) # shape:[batch_size,1]  
        torch_dataset = Data.TensorDataset(states, actions, old_probs, old_log_probs,returns, rewards, next_states, dones)
        train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=self.sgd_batch_size, shuffle=True,drop_last=False)
        for _ in range(self.k_epochs):
            for batch_idx, (states_sgd, actions_sgd, old_probs_sgd, old_log_probs_sgd, returns_sgd, rewards, next_states, dones) in enumerate(train_loader):
                values_sgd, new_log_probs_sgd, entropies = self.evaluate(states_sgd,actions_sgd)
                advantages = returns_sgd - values_sgd.detach()
                self.actor_loss = torch.mean(-new_log_probs_sgd*advantages.detach())
                # + self.entropy_coef * entropies.mean()
                self.critic_loss = torch.mean(
                    F.mse_loss(values_sgd, returns_sgd.detach()))

                ## AC algorithm
                # td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
                # td_delta = td_target - self.critic(states_sgd)
                # # log_probs = torch.log(self.actor(states_sgd)['probs'].gather(1, actions_sgd))
                # output = self.actor(states_sgd)
                # mu , sigma = output['mu'], output['sigma']
                # mean = mu * self.action_scale + self.action_bias
                # std = sigma
                # dist = Normal(mean, std)
                # log_probs = dist.log_prob(actions_sgd)
                # self.actor_loss = torch.mean(-log_probs * td_delta.detach())
                # self.critic_loss = torch.mean(
                #     F.mse_loss(self.critic(states_sgd), td_target.detach()))

                self.actor_optimizer.zero_grad()
                self.actor_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.zero_grad()
                self.critic_loss.backward()
                self.critic_optimizer.step() 
        self.update_summary()

    def _compute_returns(self, rewards, dones):
        # monte carlo estimate of state rewards
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)
        # Normalizing the rewards:
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32).unsqueeze(dim=1)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 1e-5 to avoid division by zero
        return returns