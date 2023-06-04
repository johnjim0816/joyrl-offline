import ray

class BaseCollector:
    def __init__(self, cfg, data_handler = None) -> None:
        self.cfg = cfg
        self.data_handler = data_handler
    def handle_exps_after_interact(self, interact_output, *args, **kwargs):
        exps = interact_output['exps'] # get exps from interact output
        self.data_handler.add_exps(exps) # add exps to data handler
        training_data = self.data_handler.sample_training_data() # sample training data
        return training_data
    def handle_exps_after_update(self, *args, **kwargs):
        raise NotImplementedError

class SimpleCollector(BaseCollector):
    def __init__(self, cfg, data_handler) -> None:
        super().__init__(cfg, data_handler)
    
@ray.remote
class RayCollector(BaseCollector):
    def __init__(self, cfg, data_handler) -> None:
        super().__init__(cfg, data_handler)