# import sys, os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
# curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
# parent_path = os.path.dirname(curr_path)  # parent path 
# sys.path.append(parent_path)  # add path to system path
import sys,os
import argparse,datetime,importlib,yaml,time 
import gymnasium as gym
import ray
import torch.multiprocessing as mp
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter  
from config.config import GeneralConfig, MergedConfig, DefaultConfig
from framework.collectors import SimpleCollector, RayCollector
from framework.dataserver import DataServer
from framework.interactors import SimpleInteractor, RayInteractor
from framework.learners import SimpleLearner, RayLearner
from framework.stats import SimpleStatsRecorder, SimpleLogger, RayLogger, SimpleTrajCollector
from framework.workers import Worker, SimpleTester, RayTester   

from utils.utils import save_cfgs, merge_class_attrs, all_seed,save_frames_as_gif

class Main(object):
    def __init__(self) -> None:
        self.get_default_cfg()  # get default config
        self.process_yaml_cfg()  # load yaml config
        self.merge_cfgs() # merge all configs
        self.create_dirs()  # create dirs
        self.create_loggers()  # create loggers
        # print all configs
        self.print_cfgs()
        all_seed(seed=self.general_cfg.seed)  # set seed == 0 means no seed
        self.check_resources(self.general_cfg)  # check n_workers
        self.check_sample_length(self.cfg) # check onpolicy sample length

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
    
    def print_cfgs(self):
        ''' print parameters
        '''
        def print_cfg(cfg: DefaultConfig, name = ''):
            cfg_dict = vars(cfg)
            self.logger.info(f"{name}:")
            self.logger.info(''.join(['='] * 80))
            tplt = "{:^20}\t{:^20}\t{:^20}"
            self.logger.info(tplt.format("Name", "Value", "Type"))
            for k, v in cfg_dict.items():
                if v.__class__.__name__ == 'list': # convert list to str
                    v = str(v)
                if k in ['model_dir','tb_writter']:
                    continue
                if v is None: # avoid NoneType
                    v = 'None'
                if "support" in k: # avoid ndarray
                    v = str(v[0])
                self.logger.info(tplt.format(k, v, str(type(v))))
            self.logger.info(''.join(['='] * 80))
        print_cfg(self.general_cfg,name = 'General Configs')
        print_cfg(self.algo_cfg,name = 'Algo Configs')
        print_cfg(self.env_cfg,name = 'Env Configs')

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
                self.load_yaml_cfg(self.env_cfg,load_cfg,'env_cfg')
    def merge_cfgs(self):
        ''' merge all configs
        '''
        self.cfg = MergedConfig()
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
        task_dir = f"{os.getcwd()}/tasks/{self.general_cfg.mode.capitalize()}_{self.general_cfg.mp_backend}_{env_name}_{self.general_cfg.algo_name}_{curr_time}"
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
    def create_loggers(self):
        ''' create logger
        '''
        self.logger = SimpleLogger(self.cfg.log_dir)
        self.traj_collector = SimpleTrajCollector(self.cfg.res_dir)
        if self.cfg.mp_backend == 'ray': return
        self.interact_writter = SummaryWriter(log_dir=f"{self.cfg.tb_dir}/interact")
        self.policy_writter = SummaryWriter(log_dir=f"{self.cfg.tb_dir}/model")
        
    def create_single_env(self):
        ''' create single env
        '''
        env_cfg_dic = self.env_cfg.__dict__
        kwargs = {k: v for k, v in env_cfg_dic.items() if k not in env_cfg_dic['ignore_params']}
        env = gym.make(**kwargs)
        if self.env_cfg.wrapper is not None:
            wrapper_class_path = self.env_cfg.wrapper.split('.')[:-1]
            wrapper_class_name = self.env_cfg.wrapper.split('.')[-1]
            env_wapper = __import__('.'.join(wrapper_class_path), fromlist=[wrapper_class_name])
            env = getattr(env_wapper, wrapper_class_name)(env)
        return env
    def envs_config(self):
        ''' configure environment
        '''
        # register_env(self.env_cfg.id)
        envs = [] # numbers of envs, equal to cfg.n_workers
        for _ in range(self.cfg.n_workers):
            env = self.create_single_env()
            envs.append(env)
        setattr(self.cfg, 'obs_space', envs[0].observation_space)
        setattr(self.cfg, 'action_space', envs[0].action_space)
        self.logger.info(f"obs_space: {envs[0].observation_space}, n_actions: {envs[0].action_space}")  # print info
        return envs
    def policy_config(self,cfg):
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
    def check_resources(self,cfg):
        # check cpu resources
        if cfg.__dict__.get('n_workers',None) is None: # set n_workers to 1 if not set
            setattr(cfg, 'n_workers', 1)
        if not isinstance(cfg.n_workers,int) or cfg.n_workers<=0: # n_workers must >0
            raise ValueError("the parameter 'n_workers' must >0!")
        if cfg.n_workers > mp.cpu_count() - 1:
            raise ValueError("the parameter 'n_workers' must less than total numbers of cpus on your machine!")
        # check gpu resources
        if cfg.device == "cuda" and cfg.n_learners > 1:
            raise ValueError("the parameter 'n_learners' must be 1 when using gpu!")
        if cfg.device == "cuda":
            self.n_gpus_tester = 0.05
            self.n_gpus_learner = 0.9
        else:
            self.n_gpus_tester = 0
            self.n_gpus_learner = 0
    def check_sample_length(self,cfg):
        ''' check  sample length
        '''
        onpolicy_batch_size_flag = False
        onpolicy_batch_episode_flag = False
        if not hasattr(cfg, 'batch_size'):
            setattr(self.cfg, 'batch_size', -1)
        if not hasattr(cfg, 'batch_episode'):
            setattr(self.cfg, 'batch_episode', -1)
        if cfg.buffer_type.lower().startswith('onpolicy'): # on policy
            if cfg.batch_size > 0 and cfg.batch_episode > 0:
                onpolicy_batch_episode_flag = True
            elif cfg.batch_size > 0:
                onpolicy_batch_size_flag = True
            elif cfg.batch_episode > 0:
                onpolicy_batch_episode_flag = True
            else:
                raise ValueError("the parameter 'batch_size' or 'batch_episode' must >0 when using onpolicy buffer!")
            
        n_sample_steps = cfg.batch_size if onpolicy_batch_size_flag else 1 # 1 for offpolicy
        n_sample_episodes = cfg.batch_episode if onpolicy_batch_episode_flag else float("inf") # inf for offpolicy
        setattr(self.cfg, 'n_sample_steps', n_sample_steps)
        setattr(self.cfg, 'n_sample_episodes', n_sample_episodes)
        # setattr(self.cfg, 'onpolicy_batch_size_flag', onpolicy_batch_size_flag)
        # setattr(self.cfg, 'onpolicy_batch_episode_flag', onpolicy_batch_episode_flag)
            
    def single_run(self, cfg: MergedConfig):
        ''' single process run
        '''
        envs = self.envs_config()  # configure environment
        env = envs[0] # single env
        test_env = self.create_single_env() # create single env
        policy, data_handler = self.policy_config(cfg) # configure policy and data_handler
        stats_recorder = SimpleStatsRecorder(cfg) # create stats recorder
        collector = SimpleCollector(cfg, data_handler = data_handler)
        online_tester = SimpleTester(cfg,test_env) # create online tester
        interactor = SimpleInteractor(cfg,env, stats_recorder = stats_recorder) # create interactor
        learner = SimpleLearner(cfg, policy = policy, online_tester = online_tester) # create learner
        self.logger.info(f"Start {cfg.mode}ing!") # print info
        while True:
            interactor_output = interactor.run(policy, n_steps = self.cfg.n_sample_steps, n_episodes = self.cfg.n_sample_episodes, stats_recorder = stats_recorder, logger = self.logger) # run interactor
            training_data = collector.handle_exps_after_interact(interactor_output) # get training data from collector
            learner.run(training_data, stats_recorder = stats_recorder, logger=self.logger) # train learner
            if interactor.get_task_end_flag():
                break
 
        
    def ray_run(self,cfg):
        ''' ray run
        '''
        ray.shutdown()
        ray.init(include_dashboard=True)
        envs = self.envs_config()  # configure environment
        test_env = self.create_single_env() # create single env
        self.online_tester = RayTester.options(num_gpus= self.n_gpus_tester).remote(cfg,test_env) # create online tester
        policy, data_handler = self.policy_config(cfg) # create policy and data_handler
        stats_recorder = StatsRecorder.remote(cfg) # create stats recorder
        data_server = DataServer.remote(cfg) # create data server
        ray_logger = RayLogger.remote(cfg.log_dir) # create ray logger 
        learners = []
        for i in range(cfg.n_learners):
            learner = RayLearner.options(num_gpus= self.n_gpus_learner / cfg.n_learners).remote(cfg, id = i, policy = policy,data_handler = data_handler, online_tester = self.online_tester)
            learners.append(learner)
        workers = []
        for i in range(cfg.n_workers):
            worker = Worker.remote(cfg, id = i,env = envs[i], logger = ray_logger)
            worker.set_learner_id.remote(i%cfg.n_learners)
            workers.append(worker)
        worker_tasks = [worker.run.remote(data_server = data_server,learners = learners,stats_recorder = stats_recorder) for worker in workers]
        ray.get(worker_tasks) # wait for all workers finish
        ray.shutdown() # shutdown ray

    def run(self) -> None:
        s_t = time.time()
        if self.general_cfg.mp_backend == 'ray':
            self.ray_run(self.cfg)
        else:
            self.single_run(self.cfg)
        e_t = time.time()
        self.logger.info(f"Finish {self.cfg.mode}ing! total time consumed: {e_t-s_t:.2f}s")
        save_cfgs(self.save_cfgs, self.cfg.task_dir)  # save config

if __name__ == "__main__":
    main = Main()
    main.run()
