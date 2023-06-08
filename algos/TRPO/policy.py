import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical,Normal
import torch.utils.data as Data
import numpy as np
import copy

# from common.models import MLP, Critic, ActorSoftmax
from algos.base.networks import ValueNetwork, CriticNetwork, ActorNetwork
from algos.base.policies import BasePolicy
from algos.base.buffers import BufferCreator

class Policy(BasePolicy):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.cfg = cfg
        self.obs_space = cfg.obs_space
        self.action_space = cfg.action_space
        self.gamma = cfg.gamma
        self.lmbda = cfg.lmbda  # GAE参数
        self.device = torch.device(cfg.device)
        self.action_space = cfg.action_space
        
        self.sample_count = 0
        self.train_batch_size = cfg.train_batch_size
        self.alpha = cfg.alpha
        self.kl_constraint = cfg.kl_constraint 
        self.grad_steps = cfg.grad_steps
        self.search_steps = cfg.search_steps
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
        # self.summary['scalar']['actor_loss'] = self.actor_loss.item()
        self.summary['scalar']['critic_loss'] = self.critic_loss.item()

    def create_graph(self):
        self.n_states = self.obs_space.shape[0]
        n_actions = self.action_space.n
        self.actor = ActorNetwork(self.cfg, self.state_size, self.action_space).to(self.device)
        self.critic = CriticNetwork(self.cfg, self.state_size).to(self.device)
        # self.actor = ActorSoftmax(self.n_states, n_actions, hidden_dim = self.cfg.actor_hidden_dim).to(self.device)
        # self.critic = Critic(self.n_states, 1, hidden_dim=self.cfg.critic_hidden_dim).to(self.device)
        # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)

    def create_optimizer(self):
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
            
    def get_action(self, state, mode='sample', **kwargs):
        if mode == 'sample':
            action = self.sample_action(state, **kwargs)
            # self.update_policy_transition()
        elif mode == 'predict':
            action = self.predict_action(state,**kwargs)
        else:
            raise NameError('mode must be sample or predict')
        return action
    

    def sample_action(self,state,**kwargs):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)['probs']
        # print("probs = ", probs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy().item()
    
    @torch.no_grad()
    def predict_action(self, state,**kwargs):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        probs = self.actor(state)['probs']
        dist = Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy().item()
    
    """
        Modified from codes from https://hrl.boyuai.com/chapter/2/trpo%E7%AE%97%E6%B3%95/
    """
    def hessian_product_vector(self, states, old_action_dists, vector):
        # 计算hessian矩阵和向量的乘积
        new_action_dists = torch.distributions.Categorical(self.actor(states)['probs'])
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
        log_probs = torch.log(actor(states)['probs'].gather(1, actions))
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
            new_action_dist = torch.distributions.Categorical(new_actor(states)['probs'])
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


    def learn(self, **kwargs):
        
        states, actions, next_states, rewards, dones = kwargs.get('states'), kwargs.get('actions'), kwargs.get('next_states'), kwargs.get('rewards'), kwargs.get('dones')
        state_batch = torch.tensor(np.array(states), device=self.device, dtype=torch.float32) # shape:[batch_size,n_states]
        action_batch = torch.tensor(np.array(actions), device=self.device, dtype=torch.long).unsqueeze(dim=1) # shape:[batch_size,1]
        next_state_batch = torch.tensor(np.array(next_states), device=self.device, dtype=torch.float32) # shape:[batch_size,n_states]
        reward_batch = torch.tensor(np.array(rewards), device=self.device, dtype=torch.float32).unsqueeze(dim=1) # shape:[batch_size,1]
        done_batch = torch.tensor(np.array(dones), device=self.device, dtype=torch.float32).unsqueeze(dim=1) # shape:[batch_size,1]

        # print ("state_batch = ", state_batch.shape)
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

        old_log_probs = torch.log(self.actor(state_batch)['probs'].gather(dim=1, index=action_batch)).detach()
        old_action_dist = torch.distributions.Categorical(self.actor(state_batch)['probs'].detach())
        
        self.critic_loss = F.mse_loss(self.critic(state_batch), target.detach())
        self.critic_loss = torch.mean( self.critic_loss )
        self.critic_optimizer.zero_grad()
        self.critic_loss.backward()
        self.critic_optimizer.step()
        # 更新策略函数
        self.update_policy(state_batch, action_batch, old_action_dist, old_log_probs, \
            advantage, self.grad_steps, self.search_steps)        


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

        