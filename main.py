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
import torch.multiprocessing as mp
from config.config import GeneralConfig
from common.utils import get_logger, save_results, save_cfgs, plot_rewards, merge_class_attrs, all_seed, check_n_workers, save_traj,save_frames_as_gif
from common.ray_utils import GlobalVarRecorder
from envs.register import register_env

class MergedConfig:
    def __init__(self) -> None:
        pass
class Main(object):
    def __init__(self) -> None:
        pass

    def get_default_cfg(self):
        self.general_cfg = GeneralConfig()
        self.algo_name = self.general_cfg.algo_name
        algo_mod = __import__(f"algos.{self.algo_name}.config", fromlist=['AlgoConfig'])
        self.algo_cfg = algo_mod.AlgoConfig()
        self.cfgs = {'general_cfg': self.general_cfg, 'algo_cfg': self.algo_cfg}

    def print_cfgs(self, cfg):
        ''' print parameters
        '''
        cfg_dict = vars(cfg)
        self.logger.info("Hyperparameters:")
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
        if args.yaml is not None:
            with open(args.yaml) as f:
                load_cfg = yaml.load(f, Loader=yaml.FullLoader)
                # load algo config
                self.algo_name = load_cfg['general_cfg']['algo_name']
                algo_mod = __import__(f"algos.{self.algo_name}.config",
                                      fromlist=['AlgoConfig'])  # dynamic loading of modules
                self.algo_cfg = algo_mod.AlgoConfig()
                self.cfgs = {'general_cfg': self.general_cfg, 'algo_cfg': self.algo_cfg}
                # merge config
                for cfg_type in self.cfgs:
                    if load_cfg[cfg_type] is not None:
                        for k, v in load_cfg[cfg_type].items():
                            setattr(self.cfgs[cfg_type], k, v)

    def create_dirs(self, cfg):
        curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # obtain current time
        task_dir = f"{curr_path}/tasks/{cfg.mode.capitalize()}_{cfg.env_name}_{cfg.algo_name}_{curr_time}"
        setattr(cfg, 'task_dir', task_dir)
        Path(cfg.task_dir).mkdir(parents=True, exist_ok=True)
        model_dir = f"{task_dir}/models"
        setattr(cfg, 'model_dir', model_dir)
        res_dir = f"{task_dir}/results"
        setattr(cfg, 'res_dir', res_dir)
        log_dir = f"{task_dir}/logs"
        setattr(cfg, 'log_dir', log_dir)
        traj_dir = f"{task_dir}/traj"
        setattr(cfg, 'traj_dir', traj_dir)
        video_dir = f"{task_dir}/videos"
        setattr(cfg, 'video_dir', video_dir)

    def envs_config(self, cfg):
        ''' configure environment
        '''
        register_env(cfg.env_name)
        envs = [] # numbers of envs, equal to cfg.n_workers
        for i in range(cfg.n_workers):
            if cfg.render and i == 0: # only render the first env
                env = gym.make(cfg.env_name, new_step_api=cfg.new_step_api, render_mode=cfg.render_mode)  # create env
            else:
                env = gym.make(cfg.env_name, new_step_api=cfg.new_step_api)  # create env
            if cfg.wrapper is not None:
                wrapper_class_path = cfg.wrapper.split('.')[:-1]
                wrapper_class_name = cfg.wrapper.split('.')[-1]
                env_wapper = __import__('.'.join(wrapper_class_path), fromlist=[wrapper_class_name])
                env = getattr(env_wapper, wrapper_class_name)(env, new_step_api=cfg.new_step_api)
            envs.append(env)
        try:  # state dimension
            n_states = envs[0].observation_space.n  # print(hasattr(env.observation_space, 'n'))
        except AttributeError:
            n_states = envs[0].observation_space.shape[0]  # print(hasattr(env.observation_space, 'shape'))
        try:
            n_actions = envs[0].action_space.n  # action dimension
        except AttributeError:
            n_actions = envs[0].action_space.shape[0]
            self.logger.info(f"action_bound: {abs(envs[0].action_space.low[0])}")
            setattr(cfg, 'action_bound', abs(envs[0].action_space.low[0]))
        setattr(cfg, 'action_space', envs[0].action_space)
        self.logger.info(f"n_states: {n_states}, n_actions: {n_actions}")  # print info
        # update to cfg paramters
        setattr(cfg, 'n_states', n_states)
        setattr(cfg, 'n_actions', n_actions)
        return envs
    
    def evaluate(self, cfg, trainer, env, agent):
        sum_eval_reward = 0
        for _ in range(cfg.eval_eps):
            _, eval_ep_reward, _ = trainer.test_one_episode(env, agent, cfg)
            sum_eval_reward += eval_ep_reward
        mean_eval_reward = sum_eval_reward / cfg.eval_eps
        return mean_eval_reward

    def single_run(self,cfg):
        ''' single process run
        '''
        envs = self.envs_config(cfg)  # configure environment
        env = envs[0]
        agent_mod = __import__(f"algos.{cfg.algo_name}.agent", fromlist=['Agent'])
        agent = agent_mod.Agent(cfg)  # create agent
        trainer_mod = __import__(f"algos.{cfg.algo_name}.trainer", fromlist=['Trainer'])
        trainer = trainer_mod.Trainer()  # create trainer
        if cfg.load_checkpoint:
            agent.load_model(f"tasks/{cfg.load_path}/models")
        self.logger.info(f"Start {cfg.mode}ing!")
        self.logger.info(f"Env: {cfg.env_name}, Algorithm: {cfg.algo_name}, Device: {cfg.device}")
        rewards = []  # record rewards for all episodes
        steps = []  # record steps for all episodes
        if cfg.mode.lower() == 'train':
            best_ep_reward = -float('inf')
            for i_ep in range(cfg.train_eps):
                agent, res = trainer.train_one_episode(env, agent, cfg)
                ep_reward = res['ep_reward']
                ep_step = res['ep_step']
                self.logger.info(f"Episode: {i_ep + 1}/{cfg.train_eps}, Reward: {ep_reward:.3f}, Step: {ep_step}")
                rewards.append(ep_reward)
                steps.append(ep_step)
                # for _ in range
                if (i_ep + 1) % cfg.eval_per_episode == 0:
                    mean_eval_reward = self.evaluate(cfg, trainer, env, agent)
                    if mean_eval_reward >= best_ep_reward:  # update best reward
                        self.logger.info(f"Current episode {i_ep + 1} has the best eval reward: {mean_eval_reward:.3f}")
                        best_ep_reward = mean_eval_reward
                        agent.save_model(cfg.model_dir)  # save models with best reward
            # env.close()
        elif cfg.mode.lower() == 'test':
            for i_ep in range(cfg.test_eps):
                agent, res = trainer.test_one_episode(env, agent, cfg)
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
            trajectories = {'states': [], 'actions': [], 'rewards': [], 'terminals': []}
            for i_ep in range(cfg.collect_eps):
                print ("i_ep = ", i_ep, "cfg.collect_eps = ", cfg.collect_eps)
                total_reward, ep_state, ep_action, ep_reward, ep_terminal = trainer.collect_one_episode(env, agent, cfg)
                trajectories['states'] += ep_state
                trajectories['actions'] += ep_action
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
                     title=f"{cfg.mode.lower()}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}",
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
                     title=f"{cfg.mode.lower()}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}",
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
                     title=f"{cfg.mode.lower()}ing curve of {cfg.algo_name} for {cfg.env_name} with {cfg.n_workers} {cfg.device}",
                     fpath=cfg.res_dir)

    def run(self) -> None:
        self.get_default_cfg()  # get default config
        self.process_yaml_cfg()  # process yaml config
        cfg = MergedConfig()  # merge config
        cfg = merge_class_attrs(cfg, self.cfgs['general_cfg'])
        cfg = merge_class_attrs(cfg, self.cfgs['algo_cfg'])
        self.create_dirs(cfg)  # create dirs
        self.logger = get_logger(cfg.log_dir)  # create the logger
        self.print_cfgs(cfg)  # print the configuration
        all_seed(seed=cfg.seed)  # set seed == 0 means no seed
        check_n_workers(cfg)  # check n_workers
        if cfg.n_workers == 1:
            self.single_run(cfg)
        else:
            if cfg.mp_backend == 'mp':
                self.multi_run(cfg)
            else:
                self.ray_run(cfg)


if __name__ == "__main__":
    main = Main()
    main.run()
