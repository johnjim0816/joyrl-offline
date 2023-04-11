class AlgoConfig:
    '''算法超参数类
    '''
    def __init__(self) -> None:
        self.policy_type = 'Gaussian' # 策略类型                        
        self.lr = 1e-3 # 学习率 # 3e-4                                     
        self.gamma = 0.99 # 折扣因子                                       
        self.tau = 0.005 # 软更新因子                                      
        self.alpha = 0.1 # 温度参数 α 决定信息熵相对于奖励的重要性 # 0.1   
        self.automatic_entropy_tuning = False # 自动调整 α                  
        self.batch_size = 64 # 批次大小 # 256                                
        self.hidden_dim = 64 # 隐藏层维度 # 256    
        self.n_epochs = 1 # 回合数                 
        self.start_steps = 10000 # 利用前的探索步数                
        self.target_update_fre = 1 # 更新目标网络的时间间隔  
        self.buffer_size = 1000000 # 经验回放池的大小           