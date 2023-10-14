from config.general_config import DefaultConfig
     
class AlgoConfig(DefaultConfig):
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        # self.epsilon_start = 0.95 # epsilon start value
        # self.epsilon_end = 0.01 # epsilon end value
        # self.epsilon_decay = 500 # epsilon decay rate
        # self.hidden_dim = 256 # hidden_dim for MLP
        self.gamma = 0.95 # discount factor
        self.lr = 0.0001 # learning rate
        # self.buffer_size = 100000 # size of replay buffer
        # self.batch_size = 64 # batch size
        # self.target_update = 800 # target network update frequency per steps
