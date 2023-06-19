class AlgoConfig:
    '''算法超参数类
    '''
    def __init__(self) -> None:   
        self.independ_actor = True # whether to use independent actor
        # whether actor and critic share the same optimizer
        self.share_optimizer = False # if True, lr for actor and critic will be the same    
        self.action_type = "continuous" # continuous action space              
        self.critic1_lr = 1e-3 # critic1 learning rate 
        self.critic2_lr = 1e-3 # critic2 learning rate 
        self.actor_lr = 3e-4  # actor learning rate 
        self.alpha_lr = 1e-4  # alpha learning rate                 
        self.gamma = 0.99 # reward discount factor                                       
        self.tau = 0.005 # soft update factor                                     
        self.batch_size = 64 # batch size                             
        self.hidden_dim = 64 # hidden dimension  
        self.n_epochs = 1 # number of epochs                 
        self.start_steps = 10000 # number of exploration              
        self.target_update_fre = 1 # target net update frequent  
        self.buffer_size = 1000000 # buffer size 
        self.min_policy = 0 # min value for policy (for discrete action space)      
        self.buffer_type = "REPLAY_QUE" # buffer type
