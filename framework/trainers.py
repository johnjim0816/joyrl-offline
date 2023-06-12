import time

class BaseTrainer:
    def __init__(self, cfg, *args,**kwargs) -> None:
        self.cfg = cfg
        self.interactors = kwargs['interactors']
        self.learner = kwargs['learner']
        self.collector = kwargs['collector']
        self.online_tester = kwargs['online_tester']
        self.dataserver = kwargs['dataserver']
        self.stats_recorder = kwargs['stats_recorder']
        self.logger = kwargs['logger']

    def print_cfgs(self):
        ''' print parameters
        '''
        def print_cfg(cfg, name = ''):
            cfg_dict = vars(cfg)
            self.logger.info(f"{name}:")
            self.logger.info(''.join(['='] * 80))
            tplt = "{:^20}\t{:^20}\t{:^20}"
            self.logger.info(tplt.format("Name", "Value", "Type"))
            for k, v in cfg_dict.items():
                if v.__class__.__name__ == 'list': # convert list to str
                    v = str(v)
                if v is None: # avoid NoneType
                    v = 'None'
                if "support" in k: # avoid ndarray
                    v = str(v[0])
                self.logger.info(tplt.format(k, v, str(type(v))))
            self.logger.info(''.join(['='] * 80))
        print_cfg(self.cfg.general_cfg, name = 'General Configs')
        print_cfg(self.cfg.algo_cfg, name = 'Algo Configs')
        print_cfg(self.cfg.env_cfg, name = 'Env Configs')
    def run(self):
        raise NotImplementedError
    
class SimpleTrainer(BaseTrainer):
    def __init__(self, cfg, *args,**kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
        self.print_cfgs()
    def run(self):
        self.logger.info(f"Start {self.cfg.mode}ing!") # print info
        s_t = time.time() # start time
        while True:
            policy = self.learner.get_policy() # get policy from main learner
            interact_outputs = [interactor.run(policy, dataserver = self.dataserver, logger = self.logger ) for interactor in self.interactors] # run interactors
            exps_list = [interact_output['exps'] for interact_output in interact_outputs] # get exps from interact outputs
            self.collector.add_exps_list(exps_list) # handle exps after interact
            interact_summary_que_list = [interact_output['interact_summary_que'] for interact_output in interact_outputs] # get interact summary que
            for interact_summary_que in interact_summary_que_list:
                while len(interact_summary_que) > 0:
                    interact_summary_data = interact_summary_que.popleft()
                    self.stats_recorder.add_summary(interact_summary_data, writter_type = 'interact')
            if self.cfg.onpolicy_flag: 
                n_steps_per_learn = len(self.collector.get_buffer_length())
            else:
                n_steps_per_learn = self.cfg.n_steps_per_learn
            for _ in range(n_steps_per_learn):
                training_data = self.collector.get_training_data() # get training data
                learner_output = self.learner.run(training_data, dataserver = self.dataserver) # run learner
                if learner_output is not None:
                    policy_data_after_learn = learner_output.get('policy_data_after_learn', None) # get policy data after learn
                    self.collector.handle_data_after_learn(policy_data_after_learn) # handle exps after update
                    global_update_step = self.dataserver.get_update_step() # get global update step
                    if global_update_step % self.cfg.model_summary_fre == 0:
                        self.stats_recorder.add_summary((global_update_step, learner_output['policy_summary']), writter_type = 'policy')
                    if global_update_step % self.cfg.model_save_fre == 0:
                        policy = self.learner.get_policy() # get policy from main learner
                        policy.save_model(f"{self.cfg.model_dir}/{global_update_step}")
                        if self.cfg.online_eval == True: # online evaluation
                            best_flag, online_eval_reward = self.online_tester.eval(policy)
                            learn_id = self.learner.get_id()
                            self.logger.info(f"learner id: {learn_id}, update_step: {global_update_step}, online_eval_reward: {online_eval_reward:.3f}")
                            if best_flag:
                                self.logger.info(f"learner {learn_id} for current update step obtain a better online_eval_reward: {online_eval_reward:.3f}, save the best model!")
                                policy.save_model(f"{self.cfg.model_dir}/best")
            if self.dataserver.check_task_end():
                break    
        e_t = time.time() # end time
        self.logger.info(f"Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s") # print info      

class RayTrainer(BaseTrainer):
    def __init__(self,*args,**kwargs) -> None:
        pass