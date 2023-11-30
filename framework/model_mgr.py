from framework.message import Msg, MsgType
import time
import threading
import torch
from typing import Dict, List
from queue import Queue
from algos.base.policies import BasePolicy

class ModelMgr:
    def __init__(self, cfg, model_params, **kwargs) -> None:
        self.cfg = cfg
        self.dataserver = kwargs['dataserver']
        self.logger = kwargs['logger']
        self._latest_model_params_dict = {0: model_params}
        self._saved_policy_bundles: Dict[int, int] = {}
        self._saved_policy_queue = Queue(maxsize = 128)
        self._thread_save_policy = threading.Thread(target=self._save_policy)
        self._thread_save_policy.setDaemon(True)
        self.start()

    def pub_msg(self, msg: Msg):
        ''' publish message
        '''
        msg_type, msg_data = msg.type, msg.data
        if msg_type == MsgType.MODEL_MGR_PUT_MODEL_PARAMS:
            self._put_model_params(msg_data)
        elif msg_type == MsgType.MODEL_MGR_GET_MODEL_PARAMS:
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
        if update_step >= list(self._latest_model_params_dict.keys())[-1]:
            self._latest_model_params_dict[update_step] = model_params
        if update_step % self.cfg.model_save_fre == 0:
            while not self._saved_policy_queue.full(): # if queue is full, wait for 0.01s
                self._saved_policy_queue.put((update_step, model_params))
                time.sleep(0.001)
                break

    def _get_model_params(self):
        ''' get policy
        '''
        return list(self._latest_model_params_dict.values())[-1]

    def _save_policy(self):
        ''' async run
        '''
        while True:
            while not self._saved_policy_queue.empty():
                update_step, model_params = self._saved_policy_queue.get()
                torch.save(model_params, f"{self.cfg.model_dir}/{update_step}")
            global_episode = self.dataserver.pub_msg(Msg(type = MsgType.DATASERVER_GET_EPISODE))
            if global_episode >= self.cfg.max_episode:
                break
            time.sleep(1)
    

