class AlgoConfig:
    def __init__(self):
        self.continuous = False # 连续状态空间
        self.lr = 0.0003 # 学习率
        self.batch_size = 128 # batch大小
        self.train_iterations = 500 # 训练的迭代次数
        self.actor_hidden_dim = 256 # 隐藏层维度