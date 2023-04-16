from algos.BCQ.actorcritic import VAE, Actor, Critic
from torch.optim import Adam
from common.memories import ReplayBuffer
import torch.nn.functional as F
import numpy as np
import torch as t 

class Agent():
    
    def __init__(self,cfg):
        
        # state actor dims 
        self.n_states = cfg.n_states
        self.n_actions = cfg.n_actions
        
        # for calculating  action scale and action bias 
        self.action_space = cfg.action_space
        self.action_scale = (self.action_space.high-self.action_space.low)/2
        self.action_bias = (self.action_space.high+self.action_space.low)/2
        
        self.device = cfg.device
        
        # Critic:
        self.critic = Critic(self.n_states,self.n_actions,self.device,cfg.critic_hidden_dims).to(self.device)
        #### Critic target
        self.critic_target = Critic(self.n_states,self.n_actions,self.device,cfg.critic_hidden_dims).to(self.device)
        self._target_policy_copy(self.critic,self.critic_target,1.0)
        
        # Actor
        self.actor = Actor(state_dim=self.n_states,action_dim=self.n_actions,phi=cfg.phi,max_action=1.0,
                           device=self.device,hidden_dims=cfg.actor_hidden_dims).to(self.device)
        #### Actor target
        self.actor_target = Actor(state_dim=self.n_states,action_dim=self.n_actions,phi=cfg.phi,max_action=1.0,
                           device=self.device,hidden_dims=cfg.actor_hidden_dims).to(self.device)
        self._target_policy_copy(self.actor,self.actor_target,1.0)
        
        # VAE
        self.z_dim = 2*self.n_actions
        self.vae = VAE(self.n_states,self.n_actions,self.z_dim,self.device,cfg.vae_hidden_dims).to(self.device)
        
        # Optimizer
        self.critic_optimizer = Adam(self.critic.parameters(),lr=cfg.critic_lr)
        self.actor_optimizer = Adam(self.actor.parameters(),lr=cfg.actor_lr)
        self.vae_optimizer = Adam(self.vae.parameters(),lr=cfg.vae_lr)
        
        # RL parameters 
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.lmbda = cfg.lmbda # soft double Q learning 
        self.phi = cfg.phi
        
        # buffer_size
        self.buffer = ReplayBuffer(cfg.buffer_size)
        self.start_learn_buffer_size = cfg.start_learn_buffer_size
        
        # batch size
        self.batch_size = cfg.batch_size
    
    def _target_policy_copy(self,policy,policy_target,tau):
        for param, target_param in zip(policy.parameters(),policy_target.parameters()):
            target_param.data.copy_(tau*param+(1-tau)*target_param)
    
    def select_action(self,state):
        with t.no_grad():
            state = state.reshape(1,-1).repeat(100,axis=0)
            action = self.actor(state,self.vae.decode(state))
            q1 = self.critic.q1(state,action)
            index = q1.argmax(0)
        raw_action = action[index].cpu().data.numpy().flatten()
        
        return raw_action*self.action_scale + self.action_bias
        
    def train(self,iterations):
        if len(self.buffer)<self.start_learn_buffer_size:
            print("no training, because numbers of trans are very low.")
            return
        
        for iter_i in range(iterations):
            
            state, action, reward, next_state, done = self.buffer.sample(self.batch_size)
            
            # tensor
            state_t = t.as_tensor(np.array(state),dtype=t.float32).to(self.device)
            action_t = t.as_tensor(np.array(action),dtype=t.float32).to(self.device)
            reward_t = t.as_tensor(np.array(reward),dtype=t.float32).to(self.device)
            next_state_t = t.as_tensor(np.array(next_state),dtype=t.float32).to(self.device)
            done_t = t.as_tensor(np.array(done),dtype=t.float32).to(self.device)
            
            # VAE Training
            recon_action,mean,std = self.vae(state_t,action_t)
            vae_recon_loss = F.mse_loss(recon_action,action_t)
            vae_kl_loss = -0.5 * (1 + t.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = vae_recon_loss+0.5*vae_kl_loss
            
            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()
            
            # Critic Training
            ### Target Q
            with t.no_grad():
                next_state_t = t.repeat_interleave(next_state_t, 10, 0)
                target_q1, target_q2 = self.critic_target(next_state_t,self.actor_target(next_state_t,self.vae.decode(next_state_t)))
                
                target_q = self.lmbda*t.min(target_q1,target_q2) + (1-self.lmbda)*t.max(target_q1,target_q2)
                
                target_q = target_q.reshape(self.batch_size,-1).max(1)[0].reshape(-1)

                target_q = reward_t+ (1-done_t)*self.gamma*target_q
            
            q1, q2 = self.critic(state_t,action_t)
            critic_loss = F.mse_loss(q1,target_q.reshape(-1,1))+F.mse_loss(q2,target_q.reshape(-1,1))
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor Training
            actor_loss = -self.critic.q1(state_t, self.actor(state_t,self.vae.decode(state_t))).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Target Update
            self._target_policy_copy(self.critic,self.critic_target,self.tau)
            self._target_policy_copy(self.actor,self.actor_target,self.tau)
            
    def save_model(self,fpath):
        from pathlib import Path
        
        Path(fpath).mkdir(parents=True, exist_ok=True)
        t.save(self.actor.state_dict(), f"{fpath}/actor_checkpoint.pt")
        t.save(self.actor_target.state_dict(), f"{fpath}/actor_target_checkpoint.pt")
        
        t.save(self.critic.state_dict(), f"{fpath}/critic_checkpoint.pt")
        t.save(self.critic_target.state_dict(), f"{fpath}/critic_target_checkpoint.pt")
        
        t.save(self.vae.state_dict(), f"{fpath}/vae_checkpoint.pt")
        
        np.save(f"{fpath}/action_scale.npy",self.action_scale)
        np.save(f"{fpath}/action_bias.npy",self.action_bias)
        
    def load_model(self,fpath):
        
        self.actor.load_state_dict(t.load(f"{fpath}/actor_checkpoint.pt", map_location=self.device))
        self.actor_target.load_state_dict(t.load(f"{fpath}/actor_target_checkpoint.pt", map_location=self.device))
        
        self.critic.load_state_dict(t.load(f"{fpath}/critic_checkpoint.pt", map_location=self.device))
        self.critic_target.load_state_dict(t.load(f"{fpath}/critic_target_checkpoint.pt", map_location=self.device))
        
        self.vae.load_state_dict(t.load(f"{fpath}/vae_checkpoint.pt",map_location=self.device ))
        
        self.action_scale = np.load(f"{fpath}/action_scale.npy")
        self.action_bias = np.load(f"{fpath}/action_bias.npy")
        
