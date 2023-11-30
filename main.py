# import sys, os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
# curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
# parent_path = os.path.dirname(curr_path)  # parent path 
# sys.path.append(parent_path)  # add path to system path
import sys,os
import argparse,datetime,importlib,yaml,time 
import gymnasium as gym
import torch.multiprocessing as mp
from pathlib import Path
from config.general_config import GeneralConfig, MergedConfig, DefaultConfig
from framework.collector import SimpleCollector
from framework.dataserver import SimpleDataServer, RayDataServer
from framework.interactor import DummyVecInteractor
from framework.learner import SimpleLearner
from framework.recorder import SimpleStatsRecorder, RayStatsRecorder, SimpleLogger, RayLogger, SimpleTrajCollector
from framework.tester import SimpleTester, RayTester
from framework.trainer import SimpleTrainer
from framework.model_mgr import ModelMgr

from utils.utils import save_cfgs, merge_class_attrs, all_seed,save_frames_as_gif

class Main(object):
    def __init__(self) -> None:
        self.get_default_cfg()  # get default config
        self.process_yaml_cfg()  # load yaml config
        self.merge_cfgs() # merge all configs
        self.create_dirs()  # create dirs
        all_seed(seed=self.general_cfg.seed)  # set seed == 0 means no seed
        self.check_sample_length(self.cfg) # check onpolicy sample length
        
    def print_cfgs(self, logger = None):
        ''' print parameters
        '''
        def print_cfg(cfg, name = ''):
            cfg_dict = vars(cfg)
            logger.info(f"{name}:")
            logger.info(''.join(['='] * 80))
            tplt = "{:^20}\t{:^20}\t{:^20}"
            logger.info(tplt.format("Name", "Value", "Type"))
            for k, v in cfg_dict.items():
                if v.__class__.__name__ == 'list': # convert list to str
                    v = str(v)
                if v is None: # avoid NoneType
                    v = 'None'
                if "support" in k: # avoid ndarray
                    v = str(v[0])
                logger.info(tplt.format(k, v, str(type(v))))
            logger.info(''.join(['='] * 80))
        print_cfg(self.cfg.general_cfg, name = 'General Configs')
        print_cfg(self.cfg.algo_cfg, name = 'Algo Configs')
        print_cfg(self.cfg.env_cfg, name = 'Env Configs')

    def get_default_cfg(self):
        ''' get default config
        '''
        self.general_cfg = GeneralConfig() # general config
        self.algo_name = self.general_cfg.algo_name
        algo_mod = importlib.import_module(f"algos.{self.algo_name}.config") # import algo config
        self.algo_cfg = algo_mod.AlgoConfig()
        self.env_name = self.general_cfg.env_name
        env_mod = importlib.import_module(f"envs.{self.env_name}.config") # import env config
        self.env_cfg = env_mod.EnvConfig()

    def process_yaml_cfg(self):
        ''' load yaml config
        '''
        parser = argparse.ArgumentParser(description="hyperparameters")
        parser.add_argument('--yaml', default=None, type=str,
                            help='the path of config file')
        args = parser.parse_args()
        # load config from yaml file
        if args.yaml is not None:
            with open(args.yaml) as f:
                load_cfg = yaml.load(f, Loader=yaml.FullLoader)
                # load general config
                self.load_yaml_cfg(self.general_cfg,load_cfg,'general_cfg')
                # load algo config
                self.algo_name = self.general_cfg.algo_name
                algo_mod = importlib.import_module(f"algos.{self.algo_name}.config")
                self.algo_cfg = algo_mod.AlgoConfig()
                self.load_yaml_cfg(self.algo_cfg,load_cfg,'algo_cfg')
                # load env config
                self.env_name = self.general_cfg.env_name
                env_mod = importlib.import_module(f"envs.{self.env_name}.config")
                self.env_cfg = env_mod.EnvConfig()
                self.load_yaml_cfg(self.env_cfg, load_cfg, 'env_cfg')

    def merge_cfgs(self):
        ''' merge all configs
        '''
        self.cfg = MergedConfig()
        setattr(self.cfg, 'general_cfg', self.general_cfg)
        setattr(self.cfg, 'algo_cfg', self.algo_cfg)
        setattr(self.cfg, 'env_cfg', self.env_cfg)
        self.cfg = merge_class_attrs(self.cfg, self.general_cfg)
        self.cfg = merge_class_attrs(self.cfg, self.algo_cfg)
        self.cfg = merge_class_attrs(self.cfg, self.env_cfg)
        self.save_cfgs = {'general_cfg': self.general_cfg, 'algo_cfg': self.algo_cfg, 'env_cfg': self.env_cfg}

    def load_yaml_cfg(self,target_cfg: DefaultConfig,load_cfg,item):
        if load_cfg[item] is not None:
            for k, v in load_cfg[item].items():
                setattr(target_cfg, k, v)

    def create_dirs(self):
        def config_dir(dir,name = None):
            Path(dir).mkdir(parents=True, exist_ok=True)
            setattr(self.cfg, name, dir)
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
        env_name = self.env_cfg.id if self.env_cfg.id is not None else self.general_cfg.env_name
        task_dir = f"{os.getcwd()}/tasks/{self.general_cfg.mode.capitalize()}_{env_name}_{self.general_cfg.algo_name}_{curr_time}"
        dirs_dic = {
            'task_dir':task_dir,
            'model_dir':f"{task_dir}/models",
            'res_dir':f"{task_dir}/results",
            'log_dir':f"{task_dir}/logs",
            'traj_dir':f"{task_dir}/traj",
            'video_dir':f"{task_dir}/videos",
            'tb_dir':f"{task_dir}/tb_logs"
        }
        for k,v in dirs_dic.items():
            config_dir(v,name=k)

    def env_config(self):
        ''' create single env
        '''
        env_cfg_dic = self.env_cfg.__dict__
        kwargs = {k: v for k, v in env_cfg_dic.items() if k not in env_cfg_dic['ignore_params']}
        env = gym.make(**kwargs)
        setattr(self.cfg, 'obs_space', env.observation_space)
        setattr(self.cfg, 'action_space', env.action_space)
        if self.env_cfg.wrapper is not None:
            wrapper_class_path = self.env_cfg.wrapper.split('.')[:-1]
            wrapper_class_name = self.env_cfg.wrapper.split('.')[-1]
            env_wapper = __import__('.'.join(wrapper_class_path), fromlist=[wrapper_class_name])
            env = getattr(env_wapper, wrapper_class_name)(env)
        return env
    
    def policy_config(self, cfg):
        ''' configure policy and data_handler
        '''
        policy_mod = importlib.import_module(f"algos.{cfg.algo_name}.policy")
         # create agent
        data_handler_mod = importlib.import_module(f"algos.{cfg.algo_name}.data_handler")
        policy = policy_mod.Policy(cfg) 
        if cfg.load_checkpoint:
            policy.load_model(f"tasks/{cfg.load_path}/models/{cfg.load_model_step}")
        data_handler = data_handler_mod.DataHandler(cfg)
        return policy, data_handler

    def check_sample_length(self,cfg):
        ''' check  sample length
        '''
        onpolicy_flag = False
        onpolicy_batch_size_flag = False
        onpolicy_batch_episode_flag = False
        if not hasattr(cfg, 'batch_size'):
            setattr(self.cfg, 'batch_size', -1)
        if not hasattr(cfg, 'batch_episode'):
            setattr(self.cfg, 'batch_episode', -1)
        if cfg.buffer_type.lower().startswith('onpolicy'): # on policy
            onpolicy_flag = True
            if cfg.batch_size > 0 and cfg.batch_episode > 0:
                onpolicy_batch_episode_flag = True
            elif cfg.batch_size > 0:
                onpolicy_batch_size_flag = True
            elif cfg.batch_episode > 0:
                onpolicy_batch_episode_flag = True
            else:
                raise ValueError("the parameter 'batch_size' or 'batch_episode' must >0 when using onpolicy buffer!")
        if onpolicy_flag:
            n_sample_steps = cfg.batch_size if onpolicy_batch_size_flag else float("inf")
        else:
            n_sample_steps = 1 # 1 for offpolicy  
        n_sample_episodes = cfg.batch_episode if onpolicy_batch_episode_flag else float("inf") # inf for offpolicy
        setattr(self.cfg, 'onpolicy_flag', onpolicy_flag)
        setattr(self.cfg, 'n_sample_steps', n_sample_steps)
        setattr(self.cfg, 'n_sample_episodes', n_sample_episodes)

    def run(self) -> None:
        env = self.env_config() # create single env
        policy, data_handler = self.policy_config(self.cfg) # configure policy and data_handler
        dataserver = SimpleDataServer(self.cfg)
        logger = SimpleLogger(self.cfg.log_dir)
        collector = SimpleCollector(self.cfg, data_handler = data_handler)
        vec_interactor = DummyVecInteractor(self.cfg, 
                                            env = env,
                                            policy = policy,
                                            )
        learner = SimpleLearner(self.cfg, 
                                policy = policy,
                                dataserver = dataserver,
                                collector = collector
                                )
        online_tester = SimpleTester(self.cfg, 
                                    env = env,
                                    policy = policy,    
                                    logger = logger
                                    ) # create online tester
        model_mgr = ModelMgr(self.cfg, 
                            model_params = policy.get_model_params(),
                            dataserver = dataserver,
                            logger = logger
                            )
        stats_recorder = SimpleStatsRecorder(self.cfg) # create stats recorder
        self.print_cfgs(logger = logger)  # print config
        trainer = SimpleTrainer(self.cfg, 
                                dataserver = dataserver,
                                model_mgr = model_mgr,
                                vec_interactor = vec_interactor, 
                                learner = learner, 
                                collector = collector, 
                                online_tester = online_tester,
                                stats_recorder = stats_recorder, 
                                logger = logger) # create trainer
        trainer.run() # run trainer
        save_cfgs(self.save_cfgs, self.cfg.task_dir)  # save config

if __name__ == "__main__":
    main = Main()
    main.run()
