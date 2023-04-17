class AlgoConfig:
    def __init__(self):
        # 参数具体含义，请参考 docs/BCQ.md
        self.critic_hidden_dims = [400,300]
        self.actor_hidden_dims = [400,300]
        
        self.vae_hidden_dims = [750,750]
        
        self.critic_lr = 1e-3
        self.actor_lr = 1e-3
        self.vae_lr = 1e-3
        self.batch_size = 128
        
        self.gamma = 0.99
        self.tau = 0.005
        self.lmbda = 0.75
        self.phi = 0.05
        
        # train parameters
        self.iters_per_ep = 10
        self.buffer_size = int(1e5)
        self.start_learn_buffer_size = 1e3

        # parameters for collecting data
        self.collect_explore_data = True
        self.behavior_agent_name = "DDPG"
        self.behavior_agent_parameters_path = "/behave_param.yaml"
        self.behavior_policy_path = "/behaviour_models"
        self.collect_eps = 500

        
