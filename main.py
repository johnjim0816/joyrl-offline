import sys, os

os.environ[
    "KMP_DUPLICATE_LIB_OK"] = "TRUE"  # avoid "OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized."
curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path 
sys.path.append(parent_path)  # add path to system path

import argparse
import yaml

from pathlib import Path
import datetime
import gymnasium as gym
import time 
# import gym
# from gym.wrappers import RecordVideo
import ray
from ray.util.queue import Queue
import importlib
from algos.base.buffers import BufferCreator
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter  
from config.config import GeneralConfig, MergedConfig
from utils.utils import get_logger, save_results, save_cfgs, plot_rewards, merge_class_attrs, all_seed, save_traj,save_frames_as_gif
from common.ray_utils import GlobalVarRecorder
# from envs.register import register_env
from framework.stats import StatsRecorder, SimpleLogger, RayLogger
from framework.dataserver import DataServer
from framework.workers import Worker
from framework.learners import Learner

class Main(object):
    def __init__(self) -> None:
        self.get_default_cfg()  # 获取默认参数
        self.process_yaml_cfg()  # 处理yaml配置文件参数，并覆盖默认参数
        self.merge_cfgs() # 合并参数为 self.cfg
        self.create_dirs()  # 创建文件夹
        self.create_loggers()  # 创建日志记录器
        # 打印参数
        self.print_cfgs(self.general_cfg,name = 'General Configs')  
        self.print_cfgs(self.algo_cfg,name = 'Algo Configs')
        self.print_cfgs(self.env_cfg,name = 'Env Configs') 
        all_seed(seed=self.general_cfg.seed)  # set seed == 0 means no seed
        self.check_n_workers(self.general_cfg)  # 检查n_workers参数

    def get_default_cfg(self):
        ''' get default config
        '''
        self.general_cfg = GeneralConfig()
        self.algo_name = self.general_cfg.algo_name
        algo_mod = importlib.import_module(f"algos.{self.algo_name}.config")
        self.algo_cfg = algo_mod.AlgoConfig()
        self.env_name = self.general_cfg.env_name
        env_mod = importlib.import_module(f"envs.{self.env_name}.config")
        self.env_cfg = env_mod.EnvConfig()
    
    def print_cfgs(self, cfg, name = ''):
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
        ## 加载yaml参数
        if args.yaml is not None:
            with open(args.yaml) as f:
                load_cfg = yaml.load(f, Loader=yaml.FullLoader)
                ## 加载通用参数
                self.load_yaml_cfg(self.general_cfg,load_cfg,'general_cfg')
                ## 加载算法参数
                self.algo_name = self.general_cfg.algo_name
                algo_mod = importlib.import_module(f"algos.{self.algo_name}.config")
                self.algo_cfg = algo_mod.AlgoConfig()
                self.load_yaml_cfg(self.algo_cfg,load_cfg,'algo_cfg')
                ## 加载环境参数
                self.env_name = self.general_cfg.env_name
                env_mod = importlib.import_module(f"envs.{self.env_name}.config")
                self.env_cfg = env_mod.EnvConfig()
                self.load_yaml_cfg(self.env_cfg,load_cfg,'env_cfg')
    def merge_cfgs(self):
        ''' 合并参数
        '''
        self.cfg = MergedConfig()
        self.cfg = merge_class_attrs(self.cfg, self.general_cfg)
        self.cfg = merge_class_attrs(self.cfg, self.algo_cfg)
        self.cfg = merge_class_attrs(self.cfg, self.env_cfg)
        self.save_cfgs = {'general_cfg': self.general_cfg, 'algo_cfg': self.algo_cfg, 'env_cfg': self.env_cfg}

    def load_yaml_cfg(self,target_cfg,load_cfg,item):
        if load_cfg[item] is not None:
            for k, v in load_cfg[item].items():
                setattr(target_cfg, k, v)
    def create_dirs(self):
        def config_dir(dir,name = None):
            Path(dir).mkdir(parents=True, exist_ok=True)
            setattr(self.cfg, name, dir)
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
        env_name = self.env_cfg.id if self.env_cfg.id is not None else self.general_cfg.env_name
        task_dir = f"{curr_path}/tasks/{self.general_cfg.mode.capitalize()}_{env_name}_{self.general_cfg.algo_name}_{curr_time}"
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
        if self.general_cfg.mp_backend == 'ray':
            self.logger = RayLogger(self.cfg.log_dir)
        else:
            self.logger = SimpleLogger(self.cfg.log_dir)

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
            env = getattr(env_wapper, wrapper_class_name)(env, new_step_api=self.env_cfg.new_step_api)
        return env
    def envs_config(self):
        ''' configure environment
        '''
        # register_env(self.env_cfg.id)
        envs = [] # numbers of envs, equal to cfg.n_workers
        for i in range(self.cfg.n_workers):
            env = self.create_single_env()
            envs.append(env)
        setattr(self.cfg, 'obs_space', envs[0].observation_space)
        setattr(self.cfg, 'action_space', envs[0].action_space)
        self.logger.info(f"obs_space: {envs[0].observation_space}, n_actions: {envs[0].action_space}")  # print info
        return envs
    def policy_config(self,cfg):
        algo_name = cfg.algo_name
        policy_mod = importlib.import_module(f"algos.{algo_name}.policy")
         # create agent
        data_handler_mod = importlib.import_module(f"algos.{algo_name}.data_handler")
        policy = policy_mod.Policy(cfg) 
        if cfg.load_checkpoint:
            policy.load_model(f"tasks/{cfg.load_path}/models")
        data_handler = data_handler_mod.DataHandler(cfg)
        return policy, data_handler
    
    def check_n_workers(self,cfg):
        if cfg.__dict__.get('n_workers',None) is None: # set n_workers to 1 if not set
            setattr(cfg, 'n_workers', 1)
        if not isinstance(cfg.n_workers,int) or cfg.n_workers<=0: # n_workers must >0
            raise ValueError("the parameter 'n_workers' must >0!")
        if cfg.n_workers > mp.cpu_count() - 1:
            raise ValueError("the parameter 'n_workers' must less than total numbers of cpus on your machine!")
        
    def evaluate(self, cfg, trainer, env, agent):
        sum_eval_reward = 0
        for _ in range(cfg.eval_eps):
            _, res = trainer.test_one_episode(env, agent, cfg)
            sum_eval_reward += res['ep_reward']
        mean_eval_reward = sum_eval_reward / cfg.eval_eps
        return mean_eval_reward

    def single_run(self,cfg):
        ''' single process run
        '''
        envs = self.envs_config()  # configure environment
        env = envs[0]
        policy, data_handler = self.policy_config(cfg)
        i_ep , update_step, sample_count = 0, 0, 1
        self.logger.info(f"Start {cfg.mode}ing!") # print info
        while True:
            ep_reward, ep_step = 0, 0 # reward per episode, step per episode
            state, info = env.reset(seed = cfg.seed) # reset env
            while True:
                action = policy.sample_action(state,sample_count = sample_count) # sample action
                next_state, reward, terminated, truncated , info = env.step(action) # update env
                if cfg.mode.lower() == 'train':
                    data_handler.add_transition((state, action, reward, next_state, terminated, info)) # store transition
                    training_data = data_handler.sample_training_data() # get training data
                    if training_data is not None:
                        update_step += 1
                        policy.update(**training_data,update_step=update_step)
                        model_summary = policy.summary
                        for key, value in model_summary['scalar'].items():
                            self.policy_writter.add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value=value, global_step = update_step)
                state = next_state
                ep_reward += reward
                ep_step += 1
                sample_count += 1
                if terminated or (0<= cfg.max_steps <= ep_step):
                    self.logger.info(f"episode: {i_ep}, ep_reward: {ep_reward}, ep_step: {ep_step}")
                    interact_summary = {'ep_reward': ep_reward, 'ep_step': ep_step}
                    for key, value in interact_summary.items():
                        self.interact_writter.add_scalar(tag = f"{self.cfg.mode.lower()}_{key}", scalar_value=value, global_step = i_ep)
                    i_ep += 1
                    break
            if i_ep >= cfg.max_episode:
                break
            
        # algo_name = cfg.algo_name
        # agent_mod = importlib.import_module(f"algos.{algo_name}.agent")
        # agent = agent_mod.Agent(self.cfg)  # create agent
        # trainer_mod = importlib.import_module(f"algos.{algo_name}.trainer")
        # trainer = trainer_mod.Trainer()  # create trainer
        # if cfg.load_checkpoint:
        #     agent.load_model(f"tasks/{cfg.load_path}/models")
        # self.logger.info(f"Start {cfg.mode}ing!")
        # rewards = []  # record rewards for all episodes
        # steps = []  # record steps for all episodes
        # if cfg.mode.lower() == 'train':
        #     best_ep_reward = -float('inf')
        #     for i_ep in range(cfg.train_eps):
        #         agent, res = trainer.train_one_episode(env, agent, self.cfg)
        #         ep_reward = res['ep_reward']
        #         ep_step = res['ep_step']
        #         self.logger.info(f"Episode: {i_ep + 1}/{cfg.train_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}")
        #         # for key, value in res.items():
        #         #     self.tb_writter.add_scalar(tag = f"{cfg.mode.lower()}_{key}", scalar_value=value, global_step = i_ep + 1)
        #         rewards.append(ep_reward)
        #         steps.append(ep_step)
        #         # for _ in range
        #         if (i_ep + 1) % cfg.eval_per_episode == 0:
        #             mean_eval_reward = self.evaluate(self.cfg, trainer, env, agent)
        #             if mean_eval_reward >= best_ep_reward:  # update best reward
        #                 self.logger.info(f"Current episode {i_ep + 1} has the best eval reward: {mean_eval_reward:.3f}")
        #                 best_ep_reward = mean_eval_reward
        #                 agent.save_model(cfg.model_dir)  # save models with best reward
        #     # env.close()
        # elif cfg.mode.lower() == 'test':
        #     for i_ep in range(cfg.test_eps):
        #         agent, res = trainer.test_one_episode(env, agent, self.cfg)
        #         ep_reward = res['ep_reward']
        #         ep_step = res['ep_step']
        #         self.logger.info(f"Episode: {i_ep + 1}/{cfg.test_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}")
        #         rewards.append(ep_reward)
        #         steps.append(ep_step)
        #         if i_ep == 0 and cfg.render and cfg.render_mode == 'rgb_array':
        #             frames = res['ep_frames']
        #             save_frames_as_gif(frames, cfg.video_dir)
        #     agent.save_model(cfg.model_dir)  # save models
        #     env.close()
        # elif cfg.mode.lower() == 'collect':  # collect
        #     trajectories = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'terminals': []}
        #     for i_ep in range(cfg.collect_eps):
        #         print ("i_ep = ", i_ep, "cfg.collect_eps = ", cfg.collect_eps)
        #         total_reward, ep_state, ep_action, ep_next_state, ep_reward, ep_terminal = trainer.collect_one_episode(env, agent, self.cfg)
        #         trajectories['states'] += ep_state
        #         trajectories['actions'] += ep_action
        #         trajectories['next_states'] += ep_next_state
        #         trajectories['rewards'] += ep_reward
        #         trajectories['terminals'] += ep_terminal
        #         self.logger.info(f'trajectories {i_ep + 1} collected, reward {total_reward}')
        #         rewards.append(total_reward)
        #         steps.append(cfg.max_steps)
        #     env.close()
        #     save_traj(trajectories, cfg.traj_dir)
        #     self.logger.info(f"trajectories saved to {cfg.traj_dir}")
        # self.logger.info(f"Finish {cfg.mode}ing!")
        # res_dic = {'episodes': range(len(rewards)), 'rewards': rewards, 'steps': steps}
        # save_results(res_dic, cfg.res_dir)  # save results
        # save_cfgs(self.save_cfgs, cfg.task_dir)  # save config
        # plot_rewards(rewards,
        #              title=f"{cfg.mode.lower()}ing curve on {cfg.device} of {cfg.algo_name} for {self.env_cfg.id}",
        #              fpath=cfg.res_dir)
    def ray_run(self,cfg):
        ''' ray run
        '''
        ray.shutdown()
        ray.init(include_dashboard=True)
        envs = self.envs_config()  # configure environment
        policy, data_handler = self.policy_config(cfg)
        stats_recorder = StatsRecorder.remote(cfg)
        data_server = DataServer.remote(cfg)
        self.logger = RayLogger.remote(cfg.log_dir)
        learner = Learner.remote(cfg, policy = policy,data_handler = data_handler)
        workers = []
        for i in range(cfg.n_workers):
            worker = Worker.remote(cfg,id = i,env = envs[i], logger = self.logger)
            workers.append(worker)
        worker_tasks = [worker.run.remote(data_server = data_server,learner = learner,stats_recorder = stats_recorder) for worker in workers]
        ray.get(worker_tasks)
        ray.shutdown()

    

    def run(self) -> None:

        s_t = time.time()
        if self.general_cfg.mp_backend == 'ray':
            self.ray_run(self.cfg)
        else:
            self.single_run(self.cfg)
        e_t = time.time()
        self.logger.info(f"task finished, total time consumed: {e_t-s_t:.2f}s")

if __name__ == "__main__":
    main = Main()
    main.run()
