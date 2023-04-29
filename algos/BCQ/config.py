# config.py
from config.config import DefaultConfig
class AlgoConfig(DefaultConfig):
    def __init__(self):
        # 神经网络层配置
        self.value_layers = [
            {'layer_type': 'Linear', 'layer_dim': [400], 'activation': 'ReLU'},
            {'layer_type': 'Linear', 'layer_dim': [300], 'activation': 'ReLU'},
        ]
        self.actor_hidden_layers = [
            {'layer_type': 'Linear', 'layer_dim': [400], 'activation': 'ReLU'},
            {'layer_type': 'Linear', 'layer_dim': [300], 'activation': 'ReLU'},
        ]
        self.vae_hidden_layers = [
            {'layer_type': 'Linear', 'layer_dim': [750], 'activation': 'ReLU'},
            {'layer_type': 'Linear', 'layer_dim': [750], 'activation': 'ReLU'},
        ]

        self.critic_lr = 1e-3  # critic学习率
        self.actor_lr = 1e-3  # actor学习率
        self.vae_lr = 1e-3  # VAE学习率
        self.batch_size = 128

        self.gamma = 0.99  # 奖励折扣因子
        self.tau = 0.005  # 目标网络参数的软更新参数， 在更新目标网络参数时，参数变化越小
        self.lmbda = 0.75  # soft double Q learning: target_Q = lmbda * min(q1,q2) + (1-lmbda) * max(q1,q2)
        self.phi = 0.05  # BCQ 特有的参数， 表示 action from actor 相比经验池中的action， 最大的波动范围 (Actor中使用)
        # BCQ只能用于 near-optimal dataset的学习，类似于behavior cloning,
        # 因此 phi很小， 取0.05。

        # train parameters
        self.iters_per_ep = 10