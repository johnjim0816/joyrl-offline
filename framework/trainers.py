import time
import ray
class BaseTrainer:
    def __init__(self, cfg, *args,**kwargs) -> None:
        self.cfg = cfg
        self.vec_interactor = kwargs['vec_interactor']
        self.learner = kwargs['learner']
        self.collector = kwargs['collector']
        self.online_tester = kwargs['online_tester']
        self.dataserver = kwargs['dataserver']
        self.stats_recorder = kwargs['stats_recorder']
        self.logger = kwargs['logger']

    def run(self):
        raise NotImplementedError
    
class SimpleTrainer(BaseTrainer):
    def __init__(self, cfg, *args,**kwargs) -> None:
        super().__init__(cfg, *args, **kwargs)
    def learn(self):
        n_steps_per_learn = self.collector.get_buffer_length() if self.cfg.onpolicy_flag else self.cfg.n_steps_per_learn
        for _ in range(n_steps_per_learn):
            training_data = self.collector.get_training_data() # get training data
            learner_output = self.learner.run(training_data, dataserver = self.dataserver) # run learner
            if learner_output is not None:
                policy = self.learner.get_policy() # get policy from main learner
                self.collector.handle_data_after_learn(learner_output['policy_data_after_learn']) # handle exps after update
                self.stats_recorder.add_summary([learner_output['policy_summary']], writter_type = 'policy')
                online_tester_output = self.online_tester.run(policy, dataserver = self.dataserver, logger = self.logger) # online evaluation
                if online_tester_output is not None:
                    self.stats_recorder.add_summary([online_tester_output['summary']], writter_type = 'policy')
    def run(self):
        self.logger.info(f"Start {self.cfg.mode}ing!") # print info
        s_t = time.time() # start time
        while True:
            policy = self.learner.get_policy() # get policy from main learner
            interact_outputs = self.vec_interactor.run(policy, dataserver = self.dataserver, logger = self.logger)
            self.collector.add_exps_list([interact_output['exps'] for interact_output in interact_outputs]) # handle exps after interact
            self.stats_recorder.add_summary([interact_output['interact_summary'] for interact_output in interact_outputs], writter_type = 'interact')
            if self.cfg.mode == "train": self.learn()
            if self.dataserver.check_task_end():
                break    
        e_t = time.time() # end time
        self.logger.info(f"Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s") # print info      
