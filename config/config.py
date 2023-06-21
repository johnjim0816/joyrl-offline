class DefaultConfig:
    ''' Default parameters for running
    '''
    def __init__(self) -> None:
        pass
    def print_cfg(self):
        ''' Print all parameters
        '''
        print(self.__dict__)
        
class MergedConfig:
    ''' Merge general, algorithm and environment config
    '''
    def __init__(self) -> None:
        self.general_cfg = None
        self.algo_cfg = None
        self.env_cfg = None
        
class GeneralConfig():
    ''' General parameters for running
    '''
    def __init__(self) -> None:
        # basic settings
        self.env_name = "gym" # name of environment
        self.algo_name = "DQN" # name of algorithm
        self.mode = "train" # train, test
        self.device = "cpu" # device to use
        self.seed = 0 # random seed
        self.max_episode = 100 # number of episodes for training, set -1 to keep running
        self.max_step = 200 # number of episodes for testing, set -1 means unlimited steps
        self.collect_traj = False # if collect trajectory or not
        # multiprocessing settings
        self.mp_backend = "single" # multiprocessing backend: "ray", default "single"
        self.n_workers = 1 # number of workers
        self.n_learners = 1 # number of learners if using multi-processing, default 1
        self.share_buffer = True # if all learners share the same buffer
        # online evaluation settings
        self.online_eval = False # online evaluation or not
        self.online_eval_episode = 10 # online eval episodes
        self.model_save_fre = 500 # model save frequency per update step
        # load model settings
        self.load_checkpoint = True # if load checkpoint
        self.load_path = "Train_single_CartPole-v1_DQN_20230515-211721" # path to load model
        self.load_model_step = 'best' # load model at which step
        # stats recorder settings
        self.interact_summary_fre = 1 # record interact stats per episode
        self.model_summary_fre = 1 # record update stats per update step
