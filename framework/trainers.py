import time
import ray
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
        self.print_cfgs()

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
        
    def run(self):
        self.logger.info(f"Start {self.cfg.mode}ing!") # print info
        s_t = time.time() # start time
        while True:
            policy = self.learner.get_policy() # get policy from main learner
            interact_outputs = [interactor.run(policy, dataserver = self.dataserver, logger = self.logger ) for interactor in self.interactors] # run interactors
            self.collector.add_exps_list([interact_output['exps'] for interact_output in interact_outputs]) # handle exps after interact
            self.stats_recorder.add_summary([interact_output['interact_summary'] for interact_output in interact_outputs], writter_type = 'interact')
            n_steps_per_learn = len(self.collector.get_buffer_length()) if self.cfg.onpolicy_flag else self.cfg.n_steps_per_learn
            for _ in range(n_steps_per_learn):
                training_data = self.collector.get_training_data() # get training data
                learner_output = self.learner.run(training_data, dataserver = self.dataserver) # run learner
                if learner_output is not None:
                    policy = self.learner.get_policy() # get policy from main learner
                    self.collector.handle_data_after_learn(learner_output['policy_data_after_learn']) # handle exps after update
                    self.stats_recorder.add_summary([learner_output['policy_summary']], writter_type = 'policy')
                    self.online_tester.run(policy, dataserver = self.dataserver, logger = self.logger) # online evaluation
            if self.dataserver.check_task_end():
                break    
        e_t = time.time() # end time
        self.logger.info(f"Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s") # print info      

@ray.remote
class RayTrainer(BaseTrainer):
    def __init__(self, cfg, *args,**kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)

    def print_cfgs(self):
        ''' print parameters
        '''
        def print_cfg(cfg, name = ''):
            cfg_dict = vars(cfg)
            self.logger.info.remote(f"{name}:")
            self.logger.info.remote(''.join(['='] * 80))
            tplt = "{:^20}\t{:^20}\t{:^20}"
            self.logger.info.remote(tplt.format("Name", "Value", "Type"))
            for k, v in cfg_dict.items():
                if v.__class__.__name__ == 'list': # convert list to str
                    v = str(v)
                if v is None: # avoid NoneType
                    v = 'None'
                if "support" in k: # avoid ndarray
                    v = str(v[0])
                self.logger.info.remote(tplt.format(k, v, str(type(v))))
            self.logger.info.remote(''.join(['='] * 80))
        print_cfg(self.cfg.general_cfg, name = 'General Configs')
        print_cfg(self.cfg.algo_cfg, name = 'Algo Configs')
        print_cfg(self.cfg.env_cfg, name = 'Env Configs')
    
    def run(self):
        self.logger.info.remote(f"Start {self.cfg.mode}ing!") # print info
        s_t = time.time() # start time
        while True:
            policy = ray.get(self.learner.get_policy()) # get policy from main learner
            interact_tasks = [interactor.run.remote(policy, dataserver = self.dataserver, logger = self.logger ) for interactor in self.interactors] # run interactors
            interact_outputs = ray.get(interact_tasks)
            # exps_list = ray.put([interact_output['exps'] for interact_output in interact_outputs]) # get exps from interact outputs
            exps_list = [interact_output['exps'] for interact_output in interact_outputs]
            self.collector.add_exps_list.remote(exps_list) # handle exps after interact
            summary_all_interactors = [interact_output['interact_summary'] for interact_output in interact_outputs] # get interact summary
            self.stats_recorder.add_summary.remote(summary_all_interactors, writter_type = 'interact')
            if self.cfg.onpolicy_flag: 
                n_steps_per_learn = len(self.collector.get_buffer_length())
            else:
                n_steps_per_learn = self.cfg.n_steps_per_learn
            for _ in range(n_steps_per_learn):
                training_data = ray.get(self.collector.get_training_data.remote()) # get training data
                learner_output = ray.get(self.learner.run.remote(training_data, dataserver = self.dataserver)) # run learner
                if learner_output is not None:
                    policy_data_after_learn = learner_output['policy_data_after_learn'] # get policy data after learn
                    self.collector.handle_data_after_learn.remote(policy_data_after_learn) # handle exps after update
                    summary_all_learners = [learner_output['policy_summary']] # get policy summary
                    self.stats_recorder.add_summary.remote(summary_all_learners, writter_type = 'policy')
                    global_update_step = ray.get(self.dataserver.get_update_step.remote()) # get global update step
                    if global_update_step % self.cfg.model_save_fre == 0:
                        policy = ray.get(self.learner.remote.get_policy()) # get policy from main learner
                        policy.save_model(f"{self.cfg.model_dir}/{global_update_step}")
                        if self.cfg.online_eval == True: # online evaluation
                            best_flag, online_eval_reward = self.online_tester.eval.remote(policy)
                            learner_id = self.learner.get_id()
                            self.logger.info(f"learner id: {learner_id}, update_step: {global_update_step}, online_eval_reward: {online_eval_reward:.3f}")
                            if best_flag:
                                self.logger.info(f"learner {learner_id} for current update step obtain a better online_eval_reward: {online_eval_reward:.3f}, save the best model!")
                                policy.save_model(f"{self.cfg.model_dir}/best")
            if ray.get(self.dataserver.check_task_end.remote()):
                break   
        e_t = time.time() # end time
        self.logger.info(f"Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s") # print info      