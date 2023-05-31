class EnvConfig:
    def __init__(self) -> None:
        super().__init__()
        self.id = "CartPole-v1" # 环境名称
        self.render_mode = None # render mode: None, rgb_array, human
        self.wrapper = None # 
        self.ignore_params = ["wrapper", "ignore_params"]