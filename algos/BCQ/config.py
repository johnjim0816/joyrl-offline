class AlgoConfig:
    def __init__(self):
        self.critic_hidden_dims = [400,300]
        self.actor_hidden_dims = [400,300]
        
        self.vae_hidden_dims = [750,750] # need to modified
        
        self.critic_lr = 1e-3
        self.actor_lr = 1e-3
        self.vae_lr = 1e-3
        self.batch_size = 128
        
        self.gamma = 0.99
        self.tau = 0.005
        self.lmbda = 0.75 # soft double Q learning 
        self.phi = 0.05 # BCQ 特有的参数， 表示 action 相比存在的action， 最大的变化范围
        
        # train parameters
        self.iters_per_ep = 10
        self.buffer_size = int(1e5)
        self.start_learn_buffer_size = 1e3

	# parameters for collecting data
        #self.mode = "collect"   # 在 config/config.py中已经设置
        self.collect_explore_data = True
        self.behavior_agent_name = "DDPG"
        self.behavior_agent_parameters_path = "/behave_param.yaml"  # yaml path
        self.behavior_policy_path = "/behaviour_models"  # mode="collect"的时候 不能为None
        self.collect_eps = 500
        self.traj_dir = None  # mode="train"的时候 不能为None
        
        # 在我们训练 offline RL 算法 BCQ之前，我们需要先收集足够的数据。
        # 1. 收集数据：  我们使用的 行为策略是基于ddpg agent， 因此我们需要在joyrl 中先训练好 一个 ddpg agent，然后导入
        # 2. 将数据加载进  BCQ agent的 memories中，然后进行训练。
        
