import ray
from queue import Queue
from typing import Tuple
from framework.message import Msg, MsgType

class BaseLearner:
    def __init__(self, cfg, id = 0, policy = None, *args, **kwargs) -> None:
        self.cfg = cfg
        self.id = id
        self.policy = policy
        self.collector = kwargs['collector']
        self.dataserver = kwargs['dataserver']
        self.updated_model_params_queue = Queue(maxsize = 128)
        self.global_update_step = 0

    def pub_msg(self, msg: Msg):
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.LEARNER_UPDATE_POLICY:
            model_params = msg_data
            self._put_model_params(model_params)
            self._update_policy()
        elif msg_type == MsgType.LEARNER_GET_UPDATED_MODEL_PARAMS_QUEUE:
            return self._get_updated_model_params_queue()
        else:
            raise NotImplementedError
    
    def _get_id(self):
        return self.id
    
    def _put_updated_model_params_queue(self):
        self.updated_model_params_queue.put((self.global_update_step, self._get_model_params()))

    def _get_updated_model_params_queue(self):
        res = Queue(maxsize = 128)
        while not self.updated_model_params_queue.empty():
            res.put(self.updated_model_params_queue.get())
        return res
    
    def _get_model_params(self):
        ''' get model parameters
        '''
        return self.policy.get_model_params()
    
    def _put_model_params(self, model_params):
        ''' set model parameters
        '''
        self.policy.put_model_params(model_params)

    def _update_policy(self, training_data, *args, **kwargs):
        raise NotImplementedError
    
class SimpleLearner(BaseLearner):
    def __init__(self, cfg, id = 0, policy = None, *args, **kwargs) -> None:
        super().__init__(cfg, id, policy, *args, **kwargs)

    def run(self, *args, **kwargs):
        model_mgr = kwargs['model_mgr']
        model_params = model_mgr.pub_msg(Msg(type = MsgType.MODEL_MGR_GET_MODEL_PARAMS)) # get model params
        self.policy.put_model_params(model_params)
        collector = kwargs['collector']
        training_data = collector.pub_msg(Msg(type = MsgType.COLLECTOR_GET_TRAINING_DATA)) # get training data
        if training_data is None: return
        dataserver = kwargs['dataserver']
        curr_update_step = dataserver.pub_msg(Msg(type = MsgType.DATASERVER_GET_UPDATE_STEP))
        self.policy.learn(**training_data,update_step = curr_update_step)
        dataserver.pub_msg(Msg(type = MsgType.DATASERVER_INCREASE_UPDATE_STEP))
        model_params = self.policy.get_model_params()
        model_mgr.pub_msg(Msg(type = MsgType.MODEL_MGR_PUT_MODEL_PARAMS, data = (curr_update_step, model_params)))

    # def _update_policy(self):
    #     n_steps_per_learn = self.collector.get_buffer_length() if self.cfg.onpolicy_flag else self.cfg.n_steps_per_learn
    #     for _ in range(n_steps_per_learn):
    #         training_data = self.collector.get_training_data() # get training data
    #         if training_data is None: continue
    #         self.dataserver.increase_update_step()
    #         self.global_update_step = self.dataserver.get_update_step()
    #         self.policy.learn(**training_data,update_step = self.global_update_step)
    #         self._put_updated_model_params_queue()

        # if training_data is None: return None
        # self.dataserver.increase_update_step()
        # self.global_update_step = self.dataserver.get_update_step()
        # self.policy.learn(**training_data,update_step = self.global_update_step)
        # policy_data_after_learn = self.policy.get_data_after_learn()
        # policy_summary = [(self.global_update_step,self.policy.get_summary())]
        # return {'policy_data_after_learn': policy_data_after_learn, 'policy_summary': policy_summary}
    