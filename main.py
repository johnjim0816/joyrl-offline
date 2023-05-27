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
from framework.stats import StatsRecorder, SimpleLogger, RayLogger, SimpleTrajCollector
from framework.dataserver import DataServer
from framework.workers import Worker, SimpleTester, RayTester   
from framework.learners import Learner
from utils.utils import save_cfgs, merge_class_attrs, all_seed,save_frames_as_gif

class Main(object):
    def __init__(self) -> None:
        self.get_default_cfg()  # get default config
        self.process_yaml_cfg()  # load yaml config
        self.merge_cfgs() # merge all configs
        self.create_dirs()  # create dirs
        self.create_loggers()  # create loggers
        # print all configs
        self.print_cfgs(self.general_cfg,name = 'General Configs')  
        self.print_cfgs(self.algo_cfg,name = 'Algo Configs')
        self.print_cfgs(self.env_cfg,name = 'Env Configs') 
        all_seed(seed=self.general_cfg.seed)  # set seed == 0 means no seed
        self.check_n_workers(self.general_cfg)  # check n_workers

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
    
    def print_cfgs(self, cfg: DefaultConfig, name = ''):
        ''' print parameters
        '''
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
        id = env_cfg_dic['id']
        # 查看环境是否存在
        # print(f'gym.envs.registration.registry.keys():{gym.envs.registration.registry.keys()}')
        env = gym.make(id)
        if self.env_cfg.wrapper is not None:
            wrapper_class_path = self.env_cfg.wrapper.split('.')[:-1]
            wrapper_class_name = self.env_cfg.wrapper.split('.')[-1]
            env_wapper = __import__('.'.join(wrapper_class_path), fromlist=[wrapper_class_name])
            env = getattr(env_wapper, wrapper_class_name)(env, new_step_api=self.env_cfg.new_step_api)
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
    
    def check_n_workers(self,cfg):
        ''' check n_workers
        '''
        if cfg.__dict__.get('n_workers',None) is None: # set n_workers to 1 if not set
            setattr(cfg, 'n_workers', 1)
        if not isinstance(cfg.n_workers,int) or cfg.n_workers<=0: # n_workers must >0
            raise ValueError("the parameter 'n_workers' must >0!")
        if cfg.n_workers > mp.cpu_count() - 1:
            raise ValueError("the parameter 'n_workers' must less than total numbers of cpus on your machine!")
        
    def single_run(self,cfg):
        ''' single process run
        '''
        envs = self.envs_config()  # configure environment
        env = envs[0] # single env
        test_env = self.create_single_env() # create single env
        self.online_tester = SimpleTester(cfg,test_env) # create online tester
        policy, data_handler = self.policy_config(cfg)
        i_ep , update_step, sample_count = 0, 0, 1
        self.logger.info(f"Start {cfg.mode}ing!") # print info
        while True:
            ep_reward, ep_step = 0, 0 # reward per episode, step per episode
            ep_frames = [] # frames per episode
            state, info = env.reset(seed = cfg.seed) # reset env
            if cfg.collect_traj: self.traj_collector.init_traj_cache() # init traj cache
            while True:
                if cfg.render_mode == 'rgb_array': ep_frames.append(env.render()) # render env
                get_action_mode = "sample" if cfg.mode.lower() == 'train' else "predict"
                action = policy.get_action(state,sample_count = sample_count,mode = get_action_mode) # sample action
                next_state, reward, terminated, truncated , info = env.step(action) # update env
                ep_reward += reward
                ep_step += 1
                sample_count += 1
                # store trajectories per step
                if cfg.collect_traj: self.traj_collector.add_traj_cache(state, action, reward, next_state, terminated, info)
                if cfg.mode.lower() == 'train': # train mode
                    interact_transition = {'state':state,'action':action,'reward':reward,'next_state':next_state,'done':terminated,'info':info}
                    policy_transition = policy.get_policy_transition() # get policy transition
                    transition = {**interact_transition,**policy_transition}
                    data_handler.add_transition(transition) # store transition
                    training_data = data_handler.sample_training_data() # get training data
                    if training_data is not None:
                        update_step += 1
                        policy.train(**training_data,update_step=update_step)
                        data_handler.add_data_after_train(policy.data_after_train) # add data after train
                        # save model
                        if update_step % cfg.model_save_fre == 0:
                            policy.save_model(f"{cfg.model_dir}/{update_step}")
                            if cfg.online_eval == True:
                                best_flag, online_eval_reward = self.online_tester.eval(policy)
                                self.logger.info(f"update_step: {update_step}, online_eval_reward: {online_eval_reward:.3f}")
                                if best_flag:
                                    self.logger.info(f"current update step obtain a better online_eval_reward: {online_eval_reward:.3f}, save the best model!")
                                    policy.save_model(f"{cfg.model_dir}/best")
                        model_summary = policy.summary
                        for key, value in model_summary['scalar'].items():
                            self.policy_writter.add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value=value, global_step = update_step)
                state = next_state
                if terminated or (0<= cfg.max_step <= ep_step):
                    self.logger.info(f"episode: {i_ep}, ep_reward: {ep_reward:.3f}, ep_step: {ep_step:d}")
                    interact_summary = {'reward': ep_reward, 'step': ep_step}
                    for key, value in interact_summary.items():
                        self.interact_writter.add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value=value, global_step = i_ep)
                    i_ep += 1
                    break
            task_end_flag = (i_ep >= cfg.max_episode)
            if cfg.collect_traj: self.traj_collector.store_traj(task_end_flag = task_end_flag)
            if i_ep == 1 and cfg.render_mode == 'rgb_array': save_frames_as_gif(ep_frames, cfg.video_dir) # only save the first episode
            if task_end_flag:
                break
        
    def ray_run(self,cfg):
        ''' ray run
        '''
        ray.shutdown()
        ray.init(include_dashboard=True)
        envs = self.envs_config()  # configure environment
        test_env = self.create_single_env() # create single env
        self.online_tester = RayTester.remote(cfg,test_env) # create online tester
        policy, data_handler = self.policy_config(cfg) # create policy and data_handler
        stats_recorder = StatsRecorder.remote(cfg) # create stats recorder
        data_server = DataServer.remote(cfg) # create data server
        ray_logger = RayLogger.remote(cfg.log_dir) # create ray logger 
        learners = []
        for i in range(cfg.n_learners):
            learner = Learner.remote(cfg, learner_id = i, policy = policy,data_handler = data_handler, online_tester = self.online_tester)
            learners.append(learner)
        workers = []
        for i in range(cfg.n_workers):
            worker = Worker.remote(cfg, worker_id = i,env = envs[i], logger = ray_logger)
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
