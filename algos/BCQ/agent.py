import torch as t
from torch.optim import Adam
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from common.memories import ReplayBuffer

# functions
def mlp(layer_size,activation=nn.ReLU,output_activation=None):
    layer_num = len(layer_size)
    model = nn.Sequential()
    layers = []
    for i in range(layer_num-1):
        if i==layer_num-2:
            model.add_module("linear{}".format(i), nn.Linear(layer_size[i], layer_size[i + 1]))
            if output_activation is not None:
                model.add_module("activation{}".format(i), output_activation())
        else:
            model.add_module("linear{}".format(i),nn.Linear(layer_size[i],layer_size[i+1]))
            model.add_module("activation{}".format(i),activation(inplace=False))
    return model


class VAE(nn.Module):

    def __init__(self, state_dim, action_dim, z_dim, device, hidden_dims=(700, 700)):
        super(VAE, self).__init__()
        # encoder
        # state_dim+action_dium, hidden_dims
        self.encoder = mlp((state_dim + action_dim,) + tuple(hidden_dims), output_activation=nn.ReLU)

        # mean, log_std
        self.mean = nn.Linear(hidden_dims[-1], z_dim)
        self.log_std = nn.Linear(hidden_dims[-1], z_dim)

        # decoder
        # state_dim+z_dim, reverse(hidden_dims), action_dim, final(tanh)
        self.decoder = mlp((state_dim + z_dim,) + tuple(reversed(hidden_dims)) + (action_dim,),
                           output_activation=nn.Tanh)

        self.device = device

        self.z_dim = z_dim

    def forward(self, state, action):
        state_t = t.as_tensor(state, dtype=t.float32).to(self.device)
        action_t = t.as_tensor(action, dtype=t.float32).to(self.device)

        inputs = t.cat([state_t, action_t], dim=1)

        encoder_ouput = self.encoder(inputs)

        mean = self.mean(encoder_ouput)
        log_std = self.log_std(encoder_ouput).clamp_(-4, 15)
        std = t.exp(log_std)

        z = mean + std * t.randn_like(std)

        decoder_ouput = self.decode(state_t, z)

        return decoder_ouput, mean, std

    def decode(self, state, z=None):
        state_t = t.as_tensor(state, dtype=t.float32).to(self.device)
        btz = state_t.shape[0]

        if z is None:
            z_t = t.randn((btz, self.z_dim), dtype=t.float32).to(self.device).clamp(-0.5, 0.5)
        else:
            z_t = t.as_tensor(z, dtype=t.float32).to(self.device)

        decoder_inputs = t.cat([state_t, z_t], dim=1)

        return self.decoder(decoder_inputs)


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim, phi, max_action, device, hidden_dims=(256, 256)):
        super(Actor, self).__init__()

        self.device = device
        layers = (state_dim + action_dim,) + tuple(hidden_dims) + (action_dim,)

        self.nn = mlp(layers, output_activation=nn.Tanh)

        self.max_action = t.as_tensor(max_action, dtype=t.float32).to(self.device)

        self.phi = t.as_tensor(phi, dtype=t.float32).to(self.device)

    def forward(self, state, action):
        state_t = t.as_tensor(state, dtype=t.float32).to(self.device)
        action_t = t.as_tensor(action, dtype=t.float32).to(self.device)

        inputs = t.cat([state_t, action_t], dim=1)

        outputs = (self.phi * self.max_action * self.nn(inputs) + action_t).clamp_(-self.max_action, self.max_action)

        return outputs


class Critic(nn.Module):  # double Q
    # output Q1, Q2

    def __init__(self, state_dim, action_dim, device, hidden_dims=(256, 256)):
        super(Critic, self).__init__()

        layers = (state_dim + action_dim,) + tuple(hidden_dims) + (1,)

        self.q1_fn = mlp(layers)
        self.q2_fn = mlp(layers)
        self.device = device

    def forward(self, state, action):
        state_t = t.as_tensor(state, dtype=t.float32).to(self.device)
        action_t = t.as_tensor(action, dtype=t.float32).to(self.device)

        inputs = t.cat([state_t, action_t], dim=1)

        q1 = self.q1_fn(inputs)
        q2 = self.q2_fn(inputs)

        return q1, q2

    def q1(self, state, action):
        state_t = t.as_tensor(state, dtype=t.float32).to(self.device)
        action_t = t.as_tensor(action, dtype=t.float32).to(self.device)

        inputs = t.cat([state_t, action_t], dim=1)

        q1 = self.q1_fn(inputs)

        return q1


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
        
