from framework.message import Msg, MsgType
import time
import threading
from queue import Queue

class PolicyMgr:
    def __init__(self, cfg, policy, **kwargs) -> None:
        self.cfg = cfg
        self.dataserver = kwargs['dataserver']
        self._latest_model_params_dict = {0: policy}
        self._save_policy_queue = Queue(maxsize = 128)
        self._thread_save_policy = threading.Thread(target=self._save_policy)
        self._thread_save_policy.setDaemon(True)

    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.POLICY_MGR_PUT_MODEL_PARAMS:
            self._put_model_params(msg_data)
        elif msg_type == MsgType.POLICY_MGR_GET_MODEL_PARAMS:
            return self._get_model_params()
        else:
            raise NotImplementedError
        
    def start(self):
        ''' start
        '''
        self._thread_save_policy.start()

    def _put_model_params(self, msg_data):
        ''' put model params
        '''
        update_step, model_params = msg_data
        if update_step >= self._latest_model_params_dict.keys()[-1]:
            self._latest_model_params_dict[update_step] = model_params
        while not self._save_policy_queue.full(): # if queue is full, wait for 0.01s
            self._save_policy_queue.put((update_step, model_params))
            time.sleep(0.01)
            break

    def _get_model_params(self):
        ''' get policy
        '''
        self._latest_model_params_dict.values()[-1]

    def _save_policy(self):
        ''' async run
        '''
        while True:
            while not self._save_policy_queue.empty():
                policy_step, policy = self._save_policy_queue.get()
                policy.save_model(f"{self.cfg.model_dir}/{policy_step}")
            global_episode = self.dataserver.DATASERVER_GET_EPISODE()
            if global_episode >= self.cfg.max_episode:
                break
            time.sleep(1)
    

