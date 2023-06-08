import ray

class BaseCollector:
    def __init__(self, cfg, data_handler = None) -> None:
        self.cfg = cfg
        self.n_learners = cfg.n_learners
        self.data_handler = data_handler
    def add_exps_list(self, exps_list):
        ''' add exps to data handler
        '''
        for exps in exps_list:
            self.data_handler.add_exps(exps) # add exps to data handler
    def get_training_data_list(self):
        training_data = self.data_handler.sample_training_data() # sample training data
        return training_data
    def handle_data_after_learn(self, policy_data_after_learn, *args, **kwargs):
        return 
    def get_buffer_length(self):
        return len(self.data_handler.buffer)

class SimpleCollector(BaseCollector):
    def __init__(self, cfg, data_handler) -> None:
        super().__init__(cfg, data_handler)
    
@ray.remote
class RayCollector(BaseCollector):
    def __init__(self, cfg, data_handler) -> None:
        super().__init__(cfg, data_handler)