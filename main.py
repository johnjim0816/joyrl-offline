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
import gym
from gym.wrappers import RecordVideo
import ray
from ray.util.queue import Queue
import importlib
import torch.multiprocessing as mp
from config.config import GeneralConfig, MergedConfig
from common.utils import get_logger, save_results, save_cfgs, plot_rewards, merge_class_attrs, all_seed, save_traj,save_frames_as_gif
from common.ray_utils import GlobalVarRecorder
from envs.register import register_env
from torch.utils.tensorboard import SummaryWriter  

class Main(object):
    def __init__(self) -> None:
        pass

    def get_default_cfg(self):
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
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
        if self.env_cfg.id is not None:
            env_name = self.env_cfg.id
        else:
            env_name = self.env_cfg.env_name
        task_dir = f"{curr_path}/tasks/{self.general_cfg.mode.capitalize()}_{env_name}_{self.general_cfg.algo_name}_{curr_time}"
        setattr(self.cfg, 'task_dir', task_dir)
        Path(task_dir).mkdir(parents=True, exist_ok=True)

        model_dir = f"{task_dir}/models"
        setattr(self.cfg, 'model_dir', model_dir)
        res_dir = f"{task_dir}/results"
        setattr(self.cfg, 'res_dir', res_dir)
        log_dir = f"{task_dir}/logs"
        setattr(self.cfg, 'log_dir', log_dir)
        traj_dir = f"{task_dir}/traj"
        setattr(self.cfg, 'traj_dir', traj_dir)
        video_dir = f"{task_dir}/videos"
        setattr(self.cfg, 'video_dir', video_dir)
        tb_dir = f"{task_dir}/tb_logs"
        setattr(self.cfg, 'tb_dir', tb_dir)
    def create_loggers(self):
        ''' create logger
        '''
        self.logger = get_logger(self.cfg.log_dir)
        self.tb_writter = SummaryWriter(log_dir=self.cfg.tb_dir)
    
    def envs_config(self):
        ''' configure environment
        '''
        register_env(self.env_cfg.id)
        envs = [] # numbers of envs, equal to cfg.n_workers
        for i in range(self.general_cfg.n_workers):
            env_cfg_dic = self.env_cfg.__dict__
            kwargs = {k: v for k, v in env_cfg_dic.items() if k not in env_cfg_dic['ignore_params']}
            env = gym.make(**kwargs)
            if self.env_cfg.wrapper is not None:
                wrapper_class_path = self.env_cfg.wrapper.split('.')[:-1]
                wrapper_class_name = self.env_cfg.wrapper.split('.')[-1]
                env_wapper = __import__('.'.join(wrapper_class_path), fromlist=[wrapper_class_name])
                env = getattr(env_wapper, wrapper_class_name)(env, new_step_api=self.env_cfg.new_step_api)
            envs.append(env)
        setattr(self.cfg, 'obs_space', envs[0].observation_space)
        setattr(self.cfg, 'action_space', envs[0].action_space)
        self.logger.info(f"obs_space: {envs[0].observation_space}, n_actions: {envs[0].action_space}")  # print info
        return envs
    
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
        algo_name = cfg.algo_name
        agent_mod = importlib.import_module(f"algos.{algo_name}.agent")
        agent = agent_mod.Agent(self.cfg)  # create agent
        trainer_mod = importlib.import_module(f"algos.{algo_name}.trainer")
        trainer = trainer_mod.Trainer()  # create trainer
        if cfg.load_checkpoint:
            agent.load_model(f"tasks/{cfg.load_path}/models")
        self.logger.info(f"Start {cfg.mode}ing!")
        rewards = []  # record rewards for all episodes
        steps = []  # record steps for all episodes
        if cfg.mode.lower() == 'train':
            best_ep_reward = -float('inf')
            for i_ep in range(cfg.train_eps):
                agent, res = trainer.train_one_episode(env, agent, self.cfg)
                ep_reward = res['ep_reward']
                ep_step = res['ep_step']
                self.logger.info(f"Episode: {i_ep + 1}/{cfg.train_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}")
                self.tb_writter.add_scalars(cfg.mode.lower(),{'ep_reward': ep_reward}, i_ep + 1)
                rewards.append(ep_reward)
                steps.append(ep_step)
                # for _ in range
                if (i_ep + 1) % cfg.eval_per_episode == 0:
                    mean_eval_reward = self.evaluate(self.cfg, trainer, env, agent)
                    if mean_eval_reward >= best_ep_reward:  # update best reward
                        self.logger.info(f"Current episode {i_ep + 1} has the best eval reward: {mean_eval_reward:.3f}")
                        best_ep_reward = mean_eval_reward
                        agent.save_model(cfg.model_dir)  # save models with best reward
            # env.close()
        elif cfg.mode.lower() == 'test':
            for i_ep in range(cfg.test_eps):
                agent, res = trainer.test_one_episode(env, agent, self.cfg)
                ep_reward = res['ep_reward']
                ep_step = res['ep_step']
                self.logger.info(f"Episode: {i_ep + 1}/{cfg.test_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}")
                rewards.append(ep_reward)
                steps.append(ep_step)
                if i_ep == 0 and cfg.render and cfg.render_mode == 'rgb_array':
                    frames = res['ep_frames']
                    save_frames_as_gif(frames, cfg.video_dir)
            agent.save_model(cfg.model_dir)  # save models
            env.close()
        elif cfg.mode.lower() == 'collect':  # collect
            trajectories = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'terminals': []}
            for i_ep in range(cfg.collect_eps):
                print ("i_ep = ", i_ep, "cfg.collect_eps = ", cfg.collect_eps)
                total_reward, ep_state, ep_action, ep_next_state, ep_reward, ep_terminal = trainer.collect_one_episode(env, agent, self.cfg)
                trajectories['states'] += ep_state
                trajectories['actions'] += ep_action
                trajectories['next_states'] += ep_next_state
                trajectories['rewards'] += ep_reward
                trajectories['terminals'] += ep_terminal
                self.logger.info(f'trajectories {i_ep + 1} collected, reward {total_reward}')
                rewards.append(total_reward)
                steps.append(cfg.max_steps)
            env.close()
            save_traj(trajectories, cfg.traj_dir)
            self.logger.info(f"trajectories saved to {cfg.traj_dir}")
        self.logger.info(f"Finish {cfg.mode}ing!")
        res_dic = {'episodes': range(len(rewards)), 'rewards': rewards, 'steps': steps}
        save_results(res_dic, cfg.res_dir)  # save results
        save_cfgs(self.cfgs, cfg.task_dir)  # save config
        plot_rewards(rewards,
                     title=f"{cfg.mode.lower()}ing curve on {cfg.device} of {cfg.algo_name} for {self.env_cfg.id}",
                     fpath=cfg.res_dir)
    
    def multi_run(self,cfg):
        ''' multi process run
        '''
        envs = self.envs_config(cfg)  # configure environment
        agent_mod = __import__(f"algos.{cfg.algo_name}.agent", fromlist=['Agent'])
        share_agent = agent_mod.Agent(cfg,is_share_agent = True)  # create agent
        local_agents = [agent_mod.Agent(cfg) for _ in range(cfg.n_workers)]
        worker_mod = __import__(f"algos.{cfg.algo_name}.trainer", fromlist=['Worker'])
        mp.set_start_method("spawn") # 兼容windows和unix
        if cfg.load_checkpoint:
            share_agent.load_model(f"tasks/{cfg.load_path}/models")
            for local_agent in local_agents:
                local_agent.load_model(f"tasks/{cfg.load_path}/models")
        self.logger.info(f"Start {cfg.mode}ing!")
        self.logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
        global_ep = mp.Value('i', 0)
        global_best_reward = mp.Value('d', 0.)
        global_r_que = mp.Queue()
        workers = [worker_mod.Worker(cfg,i,share_agent,envs[i],local_agents[i],global_ep=global_ep,global_r_que=global_r_que,global_best_reward=global_best_reward) for i in range(cfg.n_workers)]
        [w.start() for w in workers]
        rewards = [] # record episode reward to plot
        while True:
            r = global_r_que.get()
            if r is not None:
                rewards.append(r)
            else:
                break
        [w.join() for w in workers]
        self.logger.info(f"Finish {cfg.mode}ing!")
        res_dic = {'episodes': range(len(rewards)), 'rewards': rewards}
        save_results(res_dic, cfg.res_dir)  # save results
        save_cfgs(self.cfgs, cfg.task_dir)  # save config
        plot_rewards(rewards,
                     title=f"{cfg.mode.lower()}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.id}",
                     fpath=cfg.res_dir)
        
    def ray_run(self,cfg):
        ''' 使用Ray并行化强化学习算法
        '''
        ray.init()
        envs = self.envs_config(cfg)  # configure environment
        agent_mod = __import__(f"algos.{cfg.algo_name}.agent", fromlist=['Agent'])
        agent_mod = __import__(f"algos.{cfg.algo_name}.agent", fromlist=['ShareAgent'])
        share_agent = agent_mod.ShareAgent.remote(cfg)  # create agent
        local_agents = [agent_mod.Agent(cfg) for _ in range(cfg.n_workers)]
        worker_mod = __import__(f"algos.{cfg.algo_name}.trainer", fromlist=['WorkerRay'])
        if cfg.load_checkpoint:
            ray.get(share_agent.load_model.remote(f"tasks/{cfg.load_path}/models"))
            for local_agent in local_agents:
                local_agent.load_model(f"tasks/{cfg.load_path}/models")
        self.logger.info(f"Start {cfg.mode}ing!")
        self.logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
        global_r_que = Queue()
        # print(f'cfg.n_workers:{cfg.n_workers}')
        global_var_recorder = GlobalVarRecorder.remote() # 全局变量记录器
        ray_workers = [worker_mod.WorkerRay.remote(cfg, i, share_agent, envs[i], local_agents[i], global_r_que,global_data = global_var_recorder) for i in range(cfg.n_workers)]
        task_ids = [w.run.remote() for w in ray_workers]
        # 等待所有任务完成, 注意：ready_ids, task_ids变量不能随意改。
        while len(task_ids) > 0:
            ready_ids, task_ids = ray.wait(task_ids)
        rewards = []
        global_r_que_length = len(global_r_que)
        for _ in range(global_r_que_length):
            rewards.append(global_r_que.get())
        # sorted_dict_list形如[{episode：reward}, {episode：reward} ...]。将{episode：reward}按照episode顺序排序
        sorted_dict_list = sorted(rewards, key=lambda x: list(x.keys())[0])
        # 取出value，形成数组
        rewards = [list(d.values())[0] for d in sorted_dict_list]

        ray.shutdown()
        self.logger.info(f"Finish {cfg.mode}ing!")
        res_dic = {'episodes': range(len(rewards)), 'rewards': rewards}
        save_results(res_dic, cfg.res_dir)  # save results
        save_cfgs(self.cfgs, cfg.task_dir)  # save config
        plot_rewards(rewards,
                     title=f"{cfg.mode.lower()}ing curve of {cfg.algo_name} for {cfg.id} with {cfg.n_workers} {cfg.device}",
                     fpath=cfg.res_dir)
    def check_n_workers(self,cfg):

        if cfg.__dict__.get('n_workers',None) is None: # set n_workers to 1 if not set
            setattr(cfg, 'n_workers', 1)
        if not isinstance(cfg.n_workers,int) or cfg.n_workers<=0: # n_workers must >0
            raise ValueError("n_workers must >0!")
        if cfg.n_workers > mp.cpu_count():
            raise ValueError("n_workers must less than total numbers of cpus on your machine!")
        if cfg.n_workers > 1 and cfg.device != 'cpu':
            raise ValueError("multi process can only support cpu!")
    def run(self) -> None:
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
        if self.general_cfg.n_workers == 1:
            self.single_run(self.cfg)
        else:
            if self.general_cfg.mp_backend == 'mp':
                self.multi_run(self.cfg)
            else:
                self.ray_run(self.cfg)


if __name__ == "__main__":
    main = Main()
    main.run()
