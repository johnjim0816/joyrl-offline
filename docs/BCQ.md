# BCQ 算法参数说明

```python
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

```

* `z_dim`: 这是 VAE中latent space的维度参数，固定为action dim的两倍，因此无需设置。


# BCQ 训练过程
BCQ算法属于offline RL，因此需要 行为智能体来收集数据， 然后根据数据进行学习，而不与环境进行交互。 
算法执行步骤：
1. **生成行为智能题模型**：使用 DDPG算法 （可以自行更换其他算法） 与环境交互，模型学习好之后将模型保存。 
2. **获取训练数据**： 主目录的config下开启 "collect" mode，采用“BCQ算法”， 将DDPG算法的model复制到 “BCQ/behaviour_models”下， 
然后在 "tasks/你的训练文档/traj"下会生成memories对应的数据。
3. **训练BCQ agent**： 主目录下的config开启 "train" mode,采用“BCQ”算法，将上一步骤生成的"traj"复制到"BCQ/traj"下， 
然后训练就可以结果。 **注意**：因为训练过程中不与环境交互，因此每次训练完智能体，我们都会test_one_episode，生成reward。


# BCQ学习

## VAE介绍

[一文理解变分自编码器（VAE）](https://zhuanlan.zhihu.com/p/64485020)
[VAE手写体识别项目实现（详细注释）从小项目通俗理解变分自编码器（Variational Autoencoder, VAE）tu](https://blog.csdn.net/weixin_40015791/article/details/89735233)

## BCQ算法介绍

1. [BCQ 张楚珩](https://zhuanlan.zhihu.com/p/136844574)
2. [（RL）BCQ](https://zhuanlan.zhihu.com/p/206489894)
3. [BCQ github code](https://github.com/sfujim/BCQ/tree/master/continuous_BCQ)
4. [Batch RL与BCQ算法](https://zhuanlan.zhihu.com/p/269475418)
