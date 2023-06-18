from config.config import DefaultConfig


class AlgoConfig(DefaultConfig):
    def __init__(self) -> None:
        # set epsilon_start=epsilon_end can obtain fixed epsilon=epsilon_end
        self.epsilon_start = 0.1  # 探索概率的初始值
        self.epsilon_end = 0.001  # 探索概率的下界值
        self.epsilon_decay = 0.995  # 探索概率的衰变因子
        self.hidden_dim = 64  # 隐含层数量
        self.gamma = 0.99  # 贴现因子，值越大，表示未来的收益占更大的比重
        self.lr = 0.0001  # 学习率
        self.buffer_size = 100000  # 经验回放池的大小
        self.batch_size = 8  # 放入模型训练的 batch 大小
        self.target_update = 4  # 同步 policy 网络和 target 网络的频率

        self.lookup_step = 10  # 采样的 step 数量
        self.min_epi_num = 16  # 用于触发训练的最小 episode 数量的阈值
        self.max_epi_len = 128  # 单个episode中采样的最大step数量
        self.max_epi_num = 100  # 存放 episode 的容量大小，也就是经验回放池大小

