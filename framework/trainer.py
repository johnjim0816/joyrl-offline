import time
from framework.message import Msg, MsgType
class BaseTrainer:
    def __init__(self, cfg, *args,**kwargs) -> None:
        self.cfg = cfg
        self.policy_mgr = kwargs['policy_mgr']
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
                policy = self.learner.POLICY_MGR_GET_MODEL_PARAMS() # get policy from main learner
                self.collector.handle_data_after_learn(learner_output['policy_data_after_learn']) # handle exps after update
                self.stats_recorder.add_summary([learner_output['policy_summary']], writter_type = 'policy')
                online_tester_output = self.online_tester.run(policy, dataserver = self.dataserver, logger = self.logger) # online evaluation
                if online_tester_output is not None:
                    self.stats_recorder.add_summary([online_tester_output['summary']], writter_type = 'policy')
    def run(self):
        self.logger.info(f"Start {self.cfg.mode}ing!") # print info
        s_t = time.time() # start time
        while True:
            # get model params
            model_params = self.policy_mgr.pub_msg(Msg(type = MsgType.POLICY_MGR_GET_MODEL_PARAMS))
            # interact with env and sample data
            interact_outputs = self.vec_interactor.pub_msg(Msg(type = MsgType.INTERACTOR_SAMPLE, data = model_params))
            # deal with sampled data
            self.collector.pub_msg(Msg(type = MsgType.COLLECTOR_PUT_EXPS, data = [interact_output['exps'] for interact_output in interact_outputs]))
            self.stats_recorder.pub_msg(Msg(type = MsgType.STATS_RECORDER_PUT_INTERACT_SUMMARY, data = [interact_output['summary'] for interact_output in interact_outputs]))
            if self.cfg.mode == "train": 
                self.learner.pub_msg(Msg(type = MsgType.LEARNER_UPDATE_POLICY, data = model_params))
                updated_model_params_queue = self.learner.pub_msg(Msg(type = MsgType.LEARNER_GET_UPDATED_MODEL_PARAMS_QUEUE))
                while not updated_model_params_queue.empty():
                    update_step, updated_model_params = updated_model_params_queue.get()
                    self.policy_mgr.pub_msg(Msg(type = MsgType.POLICY_MGR_PUT_MODEL_PARAMS, data = (update_step, updated_model_params)))
            if self.dataserver.pub_msg(Msg(type = MsgType.DATASERVER_GET_EPISODE)):
                break    
        e_t = time.time() # end time
        self.logger.info(f"Finish {self.cfg.mode}ing! Time cost: {e_t - s_t:.3f} s") # print info      
