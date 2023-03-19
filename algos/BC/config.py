class AlgoConfig:
    def __init__(self):
        self.continuous = False # continuous action space
        self.lr = 0.0003 # learning rate for actor
        self.batch_size = 128 # ppo train batch size
        self.train_iterations = 200 # the iterarions of train
        self.actor_hidden_dim = 256 # hidden dimension for actor