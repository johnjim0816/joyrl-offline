from algos.base.exps import Exp
from utils.utils import save_frames_as_gif

class BaseInteractor:
    def __init__(self, cfg, env, id = 0, *args, **kwargs) -> None:
        self.cfg = cfg
        self.id = id 
        self.seed = self.cfg.seed + self.id
        self.env = env
        self.curr_state, self.info = self.env.reset(seed = self.seed)
        self.reset_interact_summary()
        self.reset_ep_params()
    def reset_interact_summary(self):
        ''' Create interact summary
        '''
        self.interact_summary = {
            'reward': [],
            'step': []
        }
    def run(self, policy, sample_count = 0,  n_steps = float("inf"), n_episodes = float("inf"),  *args, **kwargs):
        exps = []
        run_step , run_epsiode = 0, 0
        run_sample_count = sample_count
        while True:
            run_sample_count += 1
            action = policy.get_action(self.curr_state, sample_count = run_sample_count, mode = 'sample')
            next_state, reward, terminated, truncated, info = self.env.step(action)
            interact_transition = {'state':self.curr_state,'action':action,'reward':reward,'next_state':next_state,'done':terminated,'info':info}
            policy_transition = policy.get_policy_transition()
            exps.append(Exp(**interact_transition, **policy_transition))
            self.curr_state = next_state
            self.ep_reward += reward
            self.ep_step += 1
            if terminated or self.ep_step >= self.cfg.max_step:
                self.update_interact_summary()
                self.reset_ep_params()
                self.curr_state, self.info = self.env.reset(seed = self.seed)
                run_epsiode += 1
                if run_epsiode >= n_episodes:
                    break
            run_step += 1
            if run_step >= n_steps:
                break
        output = {"exps": exps, "interact_summary": self.interact_summary, "run_step": run_step, "run_epsiode": run_epsiode}
        self.reset_interact_summary()
        return output
    def update_interact_summary(self):
        ''' Update interact summary
        '''
        self.interact_summary['reward'].append(self.ep_reward)
        self.interact_summary['step'].append(self.ep_step)
    def reset_ep_params(self):
        self.ep_reward, self.ep_step = 0, 0 # reward per episode, step per episode

class SimpleInteractor(BaseInteractor):
    def __init__(self, cfg, env, id = 0, *args, **kwargs) -> None:
        super().__init__(cfg, env, id, *args, **kwargs)
        self.sample_count = 0 # sample count
        self.episode = 0 # global episode
        self.ep_frames = [] # episode frames
    def get_task_end_flag(self):
        ''' Get interact end flag
        '''
        if self.episode >= self.cfg.max_episode:
            return True
        else:
            return False
    def run(self, policy, n_steps = float("inf"), n_episodes = float("inf"), stats_recorder = None, logger = None, *args, **kwargs):
        exps = []
        run_step , run_epsiode = 0, 0 # local run step, local run episode
        while True:
            self.sample_count += 1
            if self.cfg.render_mode == 'rgb_array' and self.episode == 0: self.ep_frames.append(self.env.render()) # render env for the first episode
            action = policy.get_action(self.curr_state, sample_count = self.sample_count, mode = 'sample')
            next_state, reward, terminated, truncated, info = self.env.step(action)
            interact_transition = {'state':self.curr_state,'action':action,'reward':reward,'next_state':next_state,'done':terminated,'info':info}
            policy_transition = policy.get_policy_transition()
            exps.append(Exp(**interact_transition, **policy_transition))
            self.curr_state = next_state
            self.ep_reward += reward
            self.ep_step += 1
            if terminated or (0 < self.cfg.max_step <= self.ep_step):
                self.episode += 1
                run_epsiode += 1
                if len(self.ep_frames)>0: 
                    save_frames_as_gif(self.ep_frames, self.cfg.video_dir) # only save the first episode
                    self.ep_frames = []
                if self.episode % self.cfg.interact_summary_fre == 0: 
                    logger.info(f"Interactor {self.id} finished episode {self.episode} with reward {self.ep_reward:.3f} in {self.ep_step} steps")
                    interact_summary = {'reward':self.ep_reward,'step':self.ep_step}
                    stats_recorder.add_summary((self.episode, interact_summary), writter_type = 'interact')
                self.reset_ep_params()
                self.curr_state, self.info = self.env.reset(seed = self.seed) # reset environment
                if run_epsiode >= n_episodes:
                    break
            run_step += 1
            if run_step >= n_steps:
                break
        output = {"exps": exps,  "run_step": run_step, "run_epsiode": run_epsiode}
        return output
    
class RayInteractor(BaseInteractor):
    def __init__(self, cfg, env, id = 0, *args, **kwargs) -> None:
        super().__init__(cfg, env, id, *args, **kwargs)
        self.sample_count = 0