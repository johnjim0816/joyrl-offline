class AlgoConfig:
    def __init__(self):
        # prepare parameters
        self.algorithm_name = 'mappo'
        self.experiment_name = 'default' # !
        self.seed = 1
        self.cuda = False
        self.cuda_deterministic = True # Make sure random seed effective by default. If set, bypass such function
        self.n_training_threads = 1 # Number of torch threads for training
        self.n_rollout_threads = 5 # Number of parallel envs for training rollouts
        self.n_eval_rollout_threads = 1 # Number of parallel envs for evaluation rollouts
        self.n_render_rollout_threads = 1 # Number of parallel envs for rendering rollouts
        self.num_env_steps = 1000000 # !! Number of environment steps to train (million)
        self.user_name = 'marl' # !

        # env parameters
        self.env_name = 'MyEnv' # !
        self.use_obs_instead_of_state = False # Whether to use global state or concatenated obs

        # replay buffer parameters
        self.episode_length = 200 # !! Max length for an episode

        # network parameters
        self.share_policy = True # Whether agent share the same policy parameters
        self.use_centralized_V = True # Whether to use centralized value function
        self.stacked_frames = 1 # Number of stacked frames
        self.use_stacked_frames = False # Whether to use stacked frames as input
        self.hidden_size = 64 # Hidden size for policy and value networks
        self.layer_N = 1 # Number of layers for policy and value networks
        self.use_ReLU = True # Whether to use ReLU as activation function
        self.use_popart = False # Whether to use popart for value normalization
        self.use_valuenorm = True # Whether to use value normalization
        self.use_feature_normalization = True # Whether to apply layernorm to the inputs
        self.use_orthogonal = True # Whether to use orthogonal initialization for weights and 0 initialization for biases
        self.gain = 0.01 # The gain of last action layer

        # recurrent parameters
        self.use_naive_recurrent_policy = False # Whether to use naive recurrent policy
        self.use_recurrent_policy = False # Whether to use recurrent policy
        self.recurrent_N = 1 # Number of recurrent layers
        self.data_chunk_length = 10 # Length of data chunks used to train a recurrent policy

        # optimizer parameters
        self.lr = 5e-4 # Learning rate
        self.critic_lr = 5e-4 # Learning rate for critic
        self.opti_eps = 1e-5 # Epsilon for RMSprop optimizer
        self.weight_decay = 0

        # ppo parameters
        self.ppo_epoch = 15 # Number of ppo epochs
        self.use_clipped_value_loss = True # Whether to use clipped value loss
        self.clip_param = 0.2 # Clipping parameter for PPO
        self.num_mini_batch = 1 # Number of mini batches for ppo
        self.entropy_coef = 0.01 # Entropy coefficient
        self.value_loss_coef = 0.5 # Value loss coefficient # ! 1.0
        self.use_max_grad_norm = True # Whether to use max grad norm
        self.max_grad_norm = 0.5 # Max grad norm # ! 10.0
        self.use_gae = True # Whether to use generalized advantage estimation
        self.gamma = 0.99 # Discount factor
        self.gae_lambda = 0.95 # Lambda for GAE
        self.use_proper_time_limits = False # Whether to consider time limits when computing returns
        self.use_huber_loss = True # Whether to use Huber loss for value function
        self.use_value_active_masks = True # Whether to use active masks for value loss
        self.use_policy_active_masks = True # Whether to use active masks for policy loss
        self.huber_delta = 10.0 # Coefficience of huber loss

        # run parameters
        self.use_linear_lr_decay = False # Whether to use linear learning rate decay

        # save parameters
        self.save_interval = 1 # Save interval for models

        # log parameters
        self.log_interval = 5 # Log printing interval

        # eval parameters
        self.use_eval = False # Whether to start evaluation along with training
        self.eval_interval = 25 # Evaluation interval
        self.eval_episodes = 32 # Number of episodes for evaluation

        # render parameters
        self.save_gifs = False # Whether to save render video for evaluation
        self.use_render = False # Whether to render the environment during training
        self.render_episodes = 5 # Number of episodes for rendering a given env
        self.ifi = 0.1 # Interval between frames for rendering

        # pretrained parameters
        self.model_dir = None # Directory for pretrained models



if __name__ == '__main__':
    config = AlgoConfig()
    print(config.algorithm_name)
