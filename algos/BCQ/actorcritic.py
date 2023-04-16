import random
import numpy as np
import torch as t 
import torch.nn as nn
import torch.optim as optim

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
    
    def __init__(self,state_dim,action_dim,z_dim,device,hidden_dims=(700,700)):
        super(VAE, self).__init__()
        # encoder 
        # state_dim+action_dium, hidden_dims
        self.encoder = mlp( (state_dim+action_dim,)+tuple(hidden_dims), output_activation=nn.ReLU)
        
        # mean, log_std
        self.mean = nn.Linear(hidden_dims[-1],z_dim)
        self.log_std = nn.Linear(hidden_dims[-1],z_dim)
        
        # decoder
        # state_dim+z_dim, reverse(hidden_dims), action_dim, final(tanh)
        self.decoder = mlp( (state_dim+z_dim,)+tuple(reversed(hidden_dims))+(action_dim,),output_activation=nn.Tanh)
        
        self.device = device
        
        self.z_dim = z_dim
        
    def forward(self,state,action):
        state_t = t.as_tensor(state,dtype=t.float32).to(self.device)
        action_t = t.as_tensor(action,dtype=t.float32).to(self.device)
        
        inputs = t.cat([state_t,action_t],dim=1)
        
        encoder_ouput = self.encoder(inputs)
        
        mean = self.mean(encoder_ouput)
        log_std = self.log_std(encoder_ouput).clamp_(-4,15)
        std = t.exp(log_std)
        
        z = mean+std*t.randn_like(std)
        
        decoder_ouput = self.decode(state_t,z)
        
        return decoder_ouput, mean, std
        
    def decode(self,state,z=None):
        state_t = t.as_tensor(state,dtype=t.float32).to(self.device)
        btz = state_t.shape[0]
        
        if z is None:
            z_t = t.randn((btz,self.z_dim),dtype=t.float32).to(self.device).clamp(-0.5,0.5)
        else:
            z_t = t.as_tensor(z,dtype=t.float32).to(self.device)
        
        decoder_inputs = t.cat([state_t,z_t],dim=1)
        
        return self.decoder(decoder_inputs)

class Actor(nn.Module):
    
    def __init__(self,state_dim,action_dim,phi,max_action,device,hidden_dims=(256,256)):
        super(Actor,self).__init__()

        self.device = device
        layers = (state_dim+action_dim,)+tuple(hidden_dims)+(action_dim,)
        
        self.nn = mlp(layers,output_activation=nn.Tanh)

        self.max_action = t.as_tensor(max_action,dtype=t.float32).to(self.device)

        self.phi = t.as_tensor(phi,dtype=t.float32).to(self.device)
        
    def forward(self,state,action):
        state_t = t.as_tensor(state,dtype=t.float32).to(self.device)
        action_t = t.as_tensor(action,dtype=t.float32).to(self.device)
        
        inputs = t.cat([state_t,action_t],dim=1)

        outputs = (self.phi*self.max_action*self.nn(inputs)+action_t).clamp_(-self.max_action,self.max_action)
        
        return outputs
        
class Critic(nn.Module): # double Q
    # output Q1, Q2 
    
    def __init__(self,state_dim,action_dim,device,hidden_dims=(256,256)):
        super(Critic,self).__init__()
        
        layers = (state_dim+action_dim,)+tuple(hidden_dims)+(1,)
        
        self.q1_fn = mlp(layers)
        self.q2_fn = mlp(layers)
        self.device = device
    
    def forward(self,state,action):
        state_t = t.as_tensor(state,dtype=t.float32).to(self.device)
        action_t = t.as_tensor(action,dtype=t.float32).to(self.device)
        
        inputs = t.cat([state_t,action_t],dim=1)
        
        q1 = self.q1_fn(inputs)
        q2 = self.q2_fn(inputs)
        
        return q1,q2
    
    def q1(self,state,action):
        state_t = t.as_tensor(state,dtype=t.float32).to(self.device)
        action_t = t.as_tensor(action,dtype=t.float32).to(self.device)
        
        inputs = t.cat([state_t,action_t],dim=1)
        
        q1 = self.q1_fn(inputs)
        
        return q1     
