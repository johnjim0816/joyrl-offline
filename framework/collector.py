import ray
from framework.message import Msg, MsgType
class BaseCollector:
    def __init__(self, cfg, data_handler = None) -> None:
        self.cfg = cfg
        self.n_learners = cfg.n_learners
        self.data_handler = data_handler
    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.COLLECTOR_PUT_EXPS:
            exps_list = msg_data
            self._put_exps(exps_list)
        elif msg_type == MsgType.COLLECTOR_GET_TRAINING_DATA:
            return self._get_training_data()
        elif msg_type == MsgType.COLLECTOR_GET_BUFFER_LENGTH:
            return self.get_buffer_length()
        else:
            raise NotImplementedError
    def _put_exps(self, exps_list):
        ''' add exps to data handler
        '''
        for exps in exps_list:
            self.data_handler.add_exps(exps) # add exps to data handler
    def _get_training_data(self):
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
