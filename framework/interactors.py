from algos.base.exps import Exp

class Interactor:
    def __init__(self, cfg, env, id = 0, *args, **kwargs) -> None:
        self.cfg = cfg
        self.id = id 
        self.seed = self.cfg.seed + self.id
        self.env = env
        self.curr_state, self.info = self.env.reset(seed = self.seed)
        self.reset_interact_summaries()
        self.reset_ep_params()
    def reset_interact_summaries(self):
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
                self.update_interact_summaries()
                self.reset_ep_params()
                self.curr_state, self.info = self.env.reset(seed = self.seed)
                run_epsiode += 1
                if run_epsiode >= n_episodes:
                    break
            run_step += 1
            if run_step >= n_steps:
                break
        output = {"exps": exps, "interact_summary": self.interact_summary}
        self.reset_interact_summaries()
        return output
    def update_interact_summaries(self):
        ''' Update interact summary
        '''
        self.interact_summary['reward'].append(self.ep_reward)
        self.interact_summary['step'].append(self.ep_step)
    def reset_ep_params(self):
        self.ep_reward, self.ep_step = 0, 0 # reward per episode, step per episode