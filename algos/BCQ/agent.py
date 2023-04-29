from algos.base.layers import create_layer, LayerConfig
from algos.base.networks import DoubleCriticNetwork  # Critic
from algos.base.agents import BaseAgent
from copy import deepcopy as dcp
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import numpy as np

class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, z_dim, hidden_layers_dict):
        super(VAE, self).__init__()

        # Build Encoder
        self.encoder_layers = nn.ModuleList()
        encode_input_size = [None, state_dim + action_dim]
        for layer_dic in hidden_layers_dict:
            if "layer_type" not in layer_dic:
                raise ValueError("layer_type must be specified in layer_dic in VAE")
            layer_cfg = LayerConfig(**layer_dic)
            layer, encode_input_size = create_layer(encode_input_size, layer_cfg)
            self.encoder_layers.append(layer)

        # Mean, log_std
        mean_layer_cfg = LayerConfig(layer_type='linear', layer_dim=[z_dim], activation='none')
        log_std_layer_cfg = LayerConfig(layer_type='linear', layer_dim=[z_dim], activation='none')

        self.mean, _ = create_layer(encode_input_size, mean_layer_cfg)
        self.log_std, _ = create_layer(encode_input_size, log_std_layer_cfg)

        # Build Decoder
        self.decoder_layers = nn.ModuleList()
        reserved_hidden_layers_dict = reversed(hidden_layers_dict)

        decode_input_size = [None, state_dim + z_dim]
        for layer_dic in reserved_hidden_layers_dict:
            if "layer_type" not in layer_dic:
                raise ValueError("layer_type must be specified in layer_dic in VAE")
            layer_cfg = LayerConfig(**layer_dic)
            layer, decode_input_size = create_layer(decode_input_size, layer_cfg)
            self.decoder_layers.append(layer)
        action_layer_config = LayerConfig(layer_type='linear', layer_dim=[action_dim], activation='tanh')
        action_layer, _ = create_layer(decode_input_size, action_layer_config)
        self.decoder_layers.append(action_layer)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.z_dim = z_dim

    def forward(self,state, action):
        encoder_output = t.cat([state, action], dim=1)

        for layer in self.encoder_layers:
            encoder_output = layer(encoder_output)

        mean = self.mean(encoder_output)
        log_std = self.log_std(encoder_output)
        std = t.exp(log_std)

        z = mean + std * t.randn_like(std)

        decode_output = self.decode(state, z)

        return decode_output, mean, std


    def decode(self, state, z=None):
        batch_size = state.shape[0]
        if z is None:
            z = t.randn((batch_size, self.z_dim)).clamp(-0.5, 0.5)

        decoder_output = t.cat([state, z], dim=1)

        for layer in self.decoder_layers:
            decoder_output = layer(decoder_output)

        return decoder_output

# Actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, phi, max_action, hidden_layers_dict):
        super(Actor, self).__init__()

        input_size = [None, state_dim + action_dim]
        self.layers = nn.ModuleList()
        for layer_dict in hidden_layers_dict:
            if "layer_type" not in layer_dict:
                raise ValueError("layer_type must be specified in layer_dict in Actor")
            layer_cfg = LayerConfig(**layer_dict)
            layer, input_size = create_layer(input_size, layer_cfg)
            self.layers.append(layer)
        action_layer_cfg = LayerConfig(layer_type='linear', layer_dim=[action_dim], activation='tanh')
        action_layer, _ = create_layer(input_size, action_layer_cfg)
        self.layers.append(action_layer)

        self.max_action = t.as_tensor(max_action, dtype=t.float32)
        self.phi = t.as_tensor(phi, dtype=t.float32)

    def forward(self, state, action):
        outputs = t.cat([state, action],dim=1)

        for layer in self.layers:
            outputs = layer(outputs)

        outputs = (self.phi * self.max_action * outputs + action).clamp_(-self.max_action, self.max_action)

        return outputs

class Agent(BaseAgent):
    def __init__(self, cfg):
        super(Agent, self).__init__(cfg)

        self.cfg = cfg
        self.obs_space = cfg.obs_space
        self.action_space = cfg.action_space
        self.device = t.device(cfg.device)
        self.gamma = cfg.gamma

        # state, action dims
        self.n_states = self.obs_space.shape[0]
        self.n_actions = self.action_space.shape[0]

        # action scale and action bias
        self.action_scale = (self.action_space.high - self.action_space.low) / 2
        self.action_bias = (self.action_space.high + self.action_space.low) / 2

        # RL parameters
        self.gamma = cfg.gamma
        self.tau = cfg.tau
        self.lmbda = cfg.lmbda
        self.phi = cfg.phi
        self.batch_size = cfg.batch_size

        # buffer : we will obtain dataset from d4rl directly.
        self.buffer = None
        self.buffer_size = 0

        self.update_step = 0
        self.create_graph()

    def create_graph(self):
        # Critic
        self.critic = DoubleCriticNetwork(self.cfg, self.n_states, self.n_actions).to(self.device)
        self.critic_target = DoubleCriticNetwork(self.cfg, self.n_states, self.n_actions).to(self.device)
        self._target_policy_copy(self.critic, self.critic_target, 1.0)

        # Actor
        self.actor = Actor(state_dim=self.n_states, action_dim=self.n_actions,
                           phi=self.phi, max_action=1.0,
                           hidden_layers_dict=self.cfg.actor_hidden_layers).to(self.device)
        self.actor_target = Actor(state_dim=self.n_states, action_dim=self.n_actions,
                                  phi=self.phi, max_action=1.0,
                                  hidden_layers_dict=self.cfg.actor_hidden_layers).to(self.device)
        self._target_policy_copy(self.actor, self.actor_target, 1.0)

        # VAE
        self.z_dim = 2 * self.n_actions
        self.vae = VAE(state_dim=self.n_states, action_dim=self.n_actions,
                       z_dim=self.z_dim,
                       hidden_layers_dict=self.cfg.vae_hidden_layers).to(self.device)

        # Optimizer
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.cfg.critic_lr)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.vae_optimizer = Adam(self.vae.parameters(), lr=self.cfg.vae_lr)

    def _target_policy_copy(self, policy, policy_target, tau):
        for param, target_param in zip(policy.parameters(), policy_target.parameters()):
            target_param.data.copy_(tau * param + (1 - tau) * target_param)

    def sample_action(self, state):
        pass
        # no online traning, so no exploration and no sample action

    def predict_action(self, state):
        state = t.as_tensor(state, dtype=t.float32).to(self.device)
        with t.no_grad():
            state = state.reshape(1, -1).repeat(100, 1)
            action = self.actor(state, self.vae.decode(state))
            q1 = self.critic.q1(state, action)
            index = q1.argmax(0)
        raw_action = action[index].cpu().data.numpy().flatten()

        return raw_action * self.action_scale + self.action_bias

    def update(self, iterations):
        assert self.buffer_size!=0,  "you have not loaded the experience in the buffer"
        vae_loss_list = []
        actor_loss_list = []
        critic_loss_list = []

        for iter_i in range(iterations):
            indxs = np.random.randint(low=0, high=self.buffer_size, size=self.batch_size)

            state = self.buffer["observations"][indxs, :]
            action = self.buffer["actions"][indxs, :]
            reward = self.buffer["rewards"][indxs]
            next_state = self.buffer["next_observations"][indxs, :]
            done = self.buffer["terminals"][indxs]

            # tensor
            state_t = t.as_tensor(np.array(state), dtype=t.float32).to(self.device)
            action_t = t.as_tensor(np.array(action), dtype=t.float32).to(self.device)
            reward_t = t.as_tensor(np.array(reward), dtype=t.float32).to(self.device)
            next_state_t = t.as_tensor(np.array(next_state), dtype=t.float32).to(self.device)
            done_t = t.as_tensor(np.array(done), dtype=t.float32).to(self.device)

            # VAE Training
            recon_action, mean, std = self.vae(state_t, action_t)
            vae_recon_loss = F.mse_loss(recon_action, action_t)
            vae_kl_loss = -0.5 * (1 + t.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = vae_recon_loss + 0.5 * vae_kl_loss

            vae_loss_list.append(dcp(vae_loss.item()))

            self.vae_optimizer.zero_grad()
            vae_loss.backward()
            self.vae_optimizer.step()

            # Critic Training
            ### Target Q
            with t.no_grad():
                next_state_t = t.repeat_interleave(next_state_t, 10, 0)
                target_q1, target_q2 = self.critic_target(next_state_t, self.actor_target(next_state_t, self.vae.decode(
                    next_state_t)))

                target_q = self.lmbda * t.min(target_q1, target_q2) + (1 - self.lmbda) * t.max(target_q1, target_q2)

                target_q = target_q.reshape(self.batch_size, -1).max(1)[0].reshape(-1)

                target_q = reward_t + (1 - done_t) * self.gamma * target_q

            q1, q2 = self.critic(state_t, action_t)
            critic_loss = F.mse_loss(q1, target_q.reshape(-1, 1)) + F.mse_loss(q2, target_q.reshape(-1, 1))

            critic_loss_list.append(dcp(critic_loss.item()))

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Actor Training
            actor_loss = -self.critic.q1(state_t, self.actor(state_t, self.vae.decode(state_t))).mean()

            actor_loss_list.append(dcp(actor_loss.item()))

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Target Update
            self._target_policy_copy(self.critic, self.critic_target, self.tau)
            self._target_policy_copy(self.actor, self.actor_target, self.tau)

        self.loss_info = {"vae_loss": np.mean(vae_loss_list),
                          "actor_loss": np.mean(actor_loss_list),
                          "critic_loss": np.mean(critic_loss_list)}
        self.update_summary()
        self.update_step += 1

    def update_summary(self):
        for key, value in self.loss_info.items():
            self.cfg.tb_writter.add_scalar(tag=f"{self.cfg.mode.lower()}_{key}_loss", scalar_value=value,
                                           global_step=self.update_step)

    def save_model(self, fpath):
        from pathlib import Path

        Path(fpath).mkdir(parents=True, exist_ok=True)
        t.save(self.actor.state_dict(), f"{fpath}/actor_checkpoint.pt")
        t.save(self.actor_target.state_dict(), f"{fpath}/actor_target_checkpoint.pt")

        t.save(self.critic.state_dict(), f"{fpath}/critic_checkpoint.pt")
        t.save(self.critic_target.state_dict(), f"{fpath}/critic_target_checkpoint.pt")

        t.save(self.vae.state_dict(), f"{fpath}/vae_checkpoint.pt")

        np.save(f"{fpath}/action_scale.npy", self.action_scale)
        np.save(f"{fpath}/action_bias.npy", self.action_bias)

    def load_model(self, fpath):
        self.actor.load_state_dict(t.load(f"{fpath}/actor_checkpoint.pt", map_location=self.device))
        self.actor_target.load_state_dict(t.load(f"{fpath}/actor_target_checkpoint.pt", map_location=self.device))

        self.critic.load_state_dict(t.load(f"{fpath}/critic_checkpoint.pt", map_location=self.device))
        self.critic_target.load_state_dict(t.load(f"{fpath}/critic_target_checkpoint.pt", map_location=self.device))

        self.vae.load_state_dict(t.load(f"{fpath}/vae_checkpoint.pt", map_location=self.device))

        self.action_scale = np.load(f"{fpath}/action_scale.npy")
        self.action_bias = np.load(f"{fpath}/action_bias.npy")
