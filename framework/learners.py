import ray
from ray.util.queue import Queue, Empty, Full
from typing import Tuple
class BaseLearner:
    def __init__(self, cfg, id = 0, policy = None,*args, **kwargs) -> None:
        self.cfg = cfg
        self.id = id
        self.policy = policy

    def get_policy(self):
        return self.policy
    def get_id(self):
        return self.id
    
    def get_model_params(self):
        ''' get model parameters
        '''
        return self.policy.get_model_params()
    def set_model_params(self,model_params):
        ''' set model parameters
        '''
        self.policy.set_model_params(model_params)

    def save_model(self):
        if self.global_update_step % self.cfg.model_save_fre == 0:
            self.policy.save_model(f"{self.cfg.model_dir}/{self.global_update_step}")

    def run(self, training_data, *args, **kwargs):
        raise NotImplementedError
    
class SimpleLearner(BaseLearner):
    def __init__(self, cfg, id = 0, policy = None, data_handler=None, online_tester=None) -> None:
        super().__init__(cfg, id, policy, data_handler, online_tester)

    def run(self, training_data, *args, **kwargs):
        if training_data is None: return None
        dataserver = kwargs['dataserver']
        dataserver.increase_update_step()
        self.global_update_step = dataserver.get_update_step()
        self.policy.learn(**training_data,update_step = self.global_update_step)
        self.save_model()
        policy_data_after_learn = self.policy.get_data_after_learn()
        policy_summary = [(self.global_update_step,self.policy.get_summary())]
        return {'policy_data_after_learn': policy_data_after_learn, 'policy_summary': policy_summary}
    
    
@ray.remote
class RayLearner(BaseLearner):
    ''' learner
    '''
    def __init__(self, cfg, id = 0, policy=None, data_handler=None, *args, **kwargs) -> None:
        super().__init__(cfg, id, policy, data_handler, *args, **kwargs)
        self.model_params_que = Queue(maxsize=128)
        self.data_server = kwargs.get('data_server', None)
    
    def run(self, training_data,  *args, **kwargs):
        ''' learn policy
        '''
        if training_data is None: return None
        dataserver = kwargs['dataserver']
        ray.get(dataserver.increase_update_step.remote())
        self.global_update_step = ray.get(dataserver.get_update_step.remote())
        self.policy.learn(**training_data,update_step = self.global_update_step)
        self.save_model()
        policy_data_after_learn = self.policy.get_data_after_learn()
        policy_summary = [(self.global_update_step,self.policy.get_summary())]
        return {'policy_data_after_learn': policy_data_after_learn, 'policy_summary': policy_summary}


