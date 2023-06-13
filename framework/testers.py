import ray
class BaseTester:
    ''' Base class for online tester
    '''
    def __init__(self, cfg, env = None) -> None:
        self.cfg = cfg
        self.env = env
        self.best_eval_reward = -float('inf')
    def run(self, policy, *args, **kwargs):
        ''' Run online tester
        '''
        ''' Evaluate policy
        '''
        raise NotImplementedError
class SimpleTester(BaseTester):
    ''' Simple online tester
    '''
    def __init__(self, cfg, env = None) -> None:
        super().__init__(cfg,env)
    def eval(self, policy, global_update_step = 0, logger = None):
        sum_eval_reward = 0
        for _ in range(self.cfg.online_eval_episode):
            state, info = self.env.reset(seed = self.cfg.seed)
            ep_reward, ep_step = 0, 0 # reward per episode, step per episode
            while True:
                action = policy.get_action(state, mode = 'predict')
                next_state, reward, terminated, truncated, info = self.env.step(action)
                state = next_state
                ep_reward += reward
                ep_step += 1
                if terminated or (0<= self.cfg.max_step <= ep_step):
                    sum_eval_reward += ep_reward
                    break
        mean_eval_reward = sum_eval_reward / self.cfg.online_eval_episode
        logger.info(f"update_step: {global_update_step}, online_eval_reward: {mean_eval_reward:.3f}")
        if mean_eval_reward >= self.best_eval_reward:
            logger.info(f"current update step obtain a better online_eval_reward: {mean_eval_reward:.3f}, save the best model!")
            policy.save_model(f"{self.cfg.model_dir}/best")
            self.best_eval_reward = mean_eval_reward
    def run(self, policy, *args, **kwargs):
        ''' Run online tester
        '''
        dataserver, logger = kwargs['dataserver'], kwargs['logger']
        global_update_step = dataserver.get_update_step() # get global update step
        if global_update_step % self.cfg.model_save_fre == 0 and self.cfg.online_eval == True:
            self.eval(policy, global_update_step = global_update_step, logger = logger)
    
@ray.remote
class RayTester(SimpleTester):
    ''' Ray online tester
    '''
    def __init__(self,cfg,env=None) -> None:
        super().__init__(cfg,env)