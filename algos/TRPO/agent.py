import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical,Normal
import torch.utils.data as Data
import numpy as np
import copy

from common.models import MLP, Critic, ActorSoftmax
from algos.base.buffers import BufferCreator
from algos.base.agents import BaseAgent

class Agent(BaseAgent):
    def __init__(self, cfg, is_share_agent = False) -> None:
        self.cfg = cfg
        self.obs_space = cfg.obs_space
        self.action_space = cfg.action_space
        self.gamma = cfg.gamma
        self.lmbda = cfg.lmbda  # GAE参数
        self.device = torch.device(cfg.device)
        self.action_space = cfg.action_space
        
        self.memory = BufferCreator(cfg)()
        self.sample_count = 0
        self.train_batch_size = cfg.train_batch_size
        self.alpha = cfg.alpha
        self.kl_constraint = cfg.kl_constraint 
        self.grad_steps = cfg.grad_steps
        self.search_steps = cfg.search_steps

        self.create_graph()

    def create_graph(self):
        self.n_states = self.obs_space.shape[0]
        n_actions = self.action_space.n
        self.actor = ActorSoftmax(self.n_states, n_actions, hidden_dim = self.cfg.actor_hidden_dim).to(self.device)
        self.critic = Critic(self.n_states, 1, hidden_dim=self.cfg.critic_hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

    def sample_action(self,state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy().item()
    @torch.no_grad()
    def predict_action(self,state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy().item()

    """
        Modified from codes from https://hrl.boyuai.com/chapter/2/trpo%E7%AE%97%E6%B3%95/
    """
    def hessian_product_vector(self, states, old_action_dists, vector):
        # 计算hessian矩阵和向量的乘积
        new_action_dists = torch.distributions.Categorical(self.actor(states))
        # 计算平均KL距离
        kl = torch.mean( torch.distributions.kl.kl_divergence(old_action_dists, new_action_dists))  
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])

        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        grad2_vector = torch.cat([grad.view(-1) for grad in grad2])
        return grad2_vector        

    def conjugate_gradient(self, grad, states, old_action_dists, grad_steps):  
        # 使用共轭梯度法
        x = torch.zeros_like(grad)
        r = grad.clone()
        p = grad.clone()
        rdotr = torch.dot(r, r)
        for i in range(grad_steps):  
            Hp = self.hessian_product_vector(states, old_action_dists, p)
            alpha = rdotr / torch.dot(p, Hp)
            x += alpha * p
            r -= alpha * Hp
            new_rdotr = torch.dot(r, r)
            if new_rdotr < 1e-10:
                break
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
        return x        

    def compute_obj(self, states, actions, advantage, old_log_probs, actor): 
        ## 计算策略目标
        log_probs = torch.log(actor(states).gather(1, actions))
        ratio = torch.exp(log_probs - old_log_probs)
        obj = torch.mean(ratio * advantage)
        return obj

    def line_search(self, states, actions, advantage, old_log_probs, old_action_dist, max_vec, search_steps):
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        old_obj = self.compute_obj(states, actions, advantage, old_log_probs, self.actor)

        for i in range(search_steps): 
            coef = self.alpha ** i
            new_para = old_para + coef * max_vec; new_actor = copy.deepcopy(self.actor)
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            new_action_dist = torch.distributions.Categorical(new_actor(states))
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dist,new_action_dist))
            new_obj = self.compute_obj(states, actions, advantage,old_log_probs, new_actor)
            if new_obj > old_obj and kl_div < self.kl_constraint:
                return new_para
        return old_para

    def update_policy(self, states, actions, old_action_dist, old_log_probs, advantage, grad_steps, search_steps):  
        ## 算共轭梯度
        surrogate_obj = self.compute_obj(states, actions, advantage, old_log_probs, self.actor)
        grads = torch.autograd.grad(surrogate_obj, self.actor.parameters())
        obj_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        descent_direction = self.conjugate_gradient(obj_grad, states, old_action_dist, grad_steps)

        Hd = self.hessian_product_vector(states, old_action_dist, descent_direction)
        max_coef = torch.sqrt(2 * self.kl_constraint / (torch.dot(descent_direction, Hd) + 1e-8))
        # 线性搜索参数
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dist, descent_direction * max_coef, search_steps)  
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())  


    def update(self, share_agent=None):
        exps = self.memory.sample(len(self.memory), sequential=True)
        # convert to tensor
        state_batch = torch.tensor(np.array([exp.state for exp in exps]), device=self.device, dtype=torch.float32) # shape:[train_batch_size,n_states]
        action_batch = torch.tensor(np.array([exp.action for exp in exps]), device=self.device, dtype=torch.long).unsqueeze(dim=1) # shape:[train_batch_size,1]
        reward_batch = torch.tensor(np.array([exp.reward for exp in exps]), device=self.device, dtype=torch.float32).unsqueeze(dim=1) # shape:[train_batch_size,1]
        next_state_batch = torch.tensor(np.array([exp.next_state for exp in exps]), device=self.device, dtype=torch.float32) # shape(batchsize,n_states)
        done_batch = torch.tensor(np.array([exp.done for exp in exps]), device=self.device, dtype=torch.float32).unsqueeze(dim=1) # shape:[train_batch_size,1]

        target = reward_batch + self.gamma * self.critic(next_state_batch) * (1 - done_batch)    
        td_delta = target - self.critic(state_batch)    

        ## calculate the advantage
        td_delta = td_delta.cpu().detach().numpy()
        advantage_list = []
        advantage_value = 0.0
        for delta in td_delta[::-1]:
            advantage_value = self.gamma * self.lmbda * advantage_value + delta
            advantage_list.append(advantage_value)
        advantage_list.reverse()
        advantage = torch.tensor(advantage_list, dtype=torch.float).to(self.device)
        # calculate the advantage

        old_log_probs = torch.log(self.actor(state_batch).gather(dim=1, index=action_batch)).detach()
        old_action_dist = torch.distributions.Categorical(self.actor(state_batch).detach())
        
        critic_loss = F.mse_loss(self.critic(state_batch), target.detach())
        critic_loss = torch.mean( critic_loss )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # 更新策略函数
        self.update_policy(state_batch, action_batch, old_action_dist, old_log_probs, \
            advantage, self.grad_steps, self.search_steps)        

    def save_model(self, fpath):
        from pathlib import Path
        Path(fpath).mkdir(parents=True, exist_ok=True)
        torch.save(self.actor.state_dict(), f"{fpath}/actor.pth")
        torch.save(self.critic.state_dict(), f"{fpath}/critic.pth")
    def load_model(self, fpath):
        actor_ckpt = torch.load(f"{fpath}/actor.pth", map_location=self.device)
        critic_ckpt = torch.load(f"{fpath}/critic.pth", map_location=self.device)
        self.actor.load_state_dict(actor_ckpt)
        self.critic.load_state_dict(critic_ckpt)
        