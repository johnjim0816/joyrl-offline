import ray
from typing import Tuple
from algos.base.exps import Exp
from config.config import MergedConfig
from utils.utils import save_frames_as_gif

class BaseInteractor:
    def __init__(self, cfg: MergedConfig, env, id = 0, *args, **kwargs) -> None:
        self.cfg = cfg
        self.id = id 
        self.env = env
        self.seed = self.cfg.seed + self.id
        self.curr_state, self.info = self.env.reset(seed = self.seed)
        self.reset_ep_params()
    def reset_summary(self):
        ''' Create interact summary
        '''
        self.summary = list()
    def update_summary(self, summary: Tuple):
        ''' Add interact summary
        '''
        self.summary.append(summary)
    def get_summary(self):
        return self.summary
    def reset_ep_params(self):
        ''' Reset episode params
        '''
        self.ep_reward, self.ep_step = 0, 0
    def run(self, policy = None, *args, **kwargs):
        raise NotImplementedError     

class SimpleInteractor(BaseInteractor):
    def __init__(self, cfg, env, id = 0, *args, **kwargs) -> None:
        super().__init__(cfg, env, id, *args, **kwargs)

    def run(self, policy = None, *args, **kwargs):
        dataserver, logger = kwargs['dataserver'], kwargs['logger']
        exps = []
        self.reset_interact_summary_que()
        run_step , run_epsiode = 0, 0 # local run step, local run episode
        while True:
            dataserver.increase_sample_count() # increase sample count
            global_sample_count = dataserver.get_sample_count() # get global sample count
            if self.cfg.render_mode == 'rgb_array' and self.episode == 0: # render env for the first episode
                dataserver.add_ep_frame(self.env.render())
            action = policy.get_action(self.curr_state, sample_count = global_sample_count)
            next_state, reward, terminated, truncated, info = self.env.step(action)
            # add transition to exps
            interact_transition = {'state':self.curr_state,'action':action,'reward':reward,'next_state':next_state,'done':terminated,'info':info}
            policy_transition = policy.get_policy_transition()
            exps.append(Exp(**interact_transition, **policy_transition))
            self.curr_state = next_state
            self.ep_reward += reward
            self.ep_step += 1
            if terminated or (0 < self.cfg.max_step <= self.ep_step):
                dataserver.increase_episode() # increase episode
                global_episode = dataserver.get_episode() # get global episode
                run_epsiode += 1
                # if len(self.ep_frames)>0: 
                #     save_frames_as_gif(self.ep_frames, self.cfg.video_dir) # only save the first episode
                #     self.ep_frames = []
                if global_episode % self.cfg.interact_summary_fre == 0 and global_episode <= self.cfg.max_episode: 
                    logger.info(f"Interactor {self.id} finished episode {global_episode} with reward {self.ep_reward:.3f} in {self.ep_step} steps")
                    interact_summary = {'reward':self.ep_reward,'step':self.ep_step}
                    self.update_summary((global_episode, interact_summary))
                self.reset_ep_params()
                self.curr_state, self.info = self.env.reset(seed = self.seed) # reset environment
                if run_epsiode >= self.cfg.n_sample_episodes:
                    break
            run_step += 1
            if run_step >= self.cfg.n_sample_steps:
                break
        output = {"exps": exps, "interact_summary": self.get_summary()}
        return output

@ray.remote    
class RayInteractor(BaseInteractor):
    def __init__(self, cfg, env, id = 0, *args, **kwargs) -> None:
        super().__init__(cfg, env, id, *args, **kwargs)
        self.data_server = kwargs.get('data_server', None)
    def run(self, policy, logger = None, stats_recorder = None, *args, **kwargs):
        exps = []
        run_step , run_epsiode = 0, 0 # local run step, local run episode
        while True:
            ray.get(self.data_server.increase_sample_count.remote())
            global_sample_count = ray.get(self.data_server.get_sample_count.remote())
            action = policy.get_action(self.curr_state, sample_count = global_sample_count, mode = 'sample')
            next_state, reward, terminated, truncated, info = self.env.step(action)
            interact_transition = {'state':self.curr_state,'action':action,'reward':reward,'next_state':next_state,'done':terminated,'info':info}
            policy_transition = policy.get_policy_transition()
            exps.append(Exp(**interact_transition, **policy_transition))
            self.curr_state = next_state
            self.ep_reward += reward
            self.ep_step += 1
            if terminated or (0 < self.cfg.max_step <= self.ep_step):
                run_epsiode += 1
                ray.get(self.data_server.increase_episode.remote())
                global_episode = ray.get(self.data_server.get_episode.remote())
                if global_episode % self.cfg.interact_summary_fre == 0:
                    logger.info.remote(f"Interactor {self.id} finished episode {global_episode} with reward {self.ep_reward:.3f} in {self.ep_step} steps")
                    interact_summary = {'reward':self.ep_reward,'step':self.ep_step}
                    stats_recorder.add_summary.remote((global_episode, interact_summary), writter_type = 'interact')
                    self.reset_ep_params()
                    self.curr_state, self.info = self.env.reset(seed = self.seed) # reset environment
                    if run_epsiode >= self.cfg.n_sample_episodes:
                        break
            run_step += 1
            if run_step >= self.cfg.n_sample_steps:
                break
        output = {"exps": exps,  "run_step": run_step, "run_epsiode": run_epsiode}
        return output
        