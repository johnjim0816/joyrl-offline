from enum import Enum
from typing import Optional, Any

class MsgType(Enum):
    # dataserver
    DATASERVER_GET_EPISODE = 0
    DATASERVER_INCREASE_EPISODE = 1
    DATASERVER_CHECK_TASK_END = 2

    # interactor
    INTERACTOR_SAMPLE = 10
    INTERACTOR_GET_SAMPLE_DATA = 11
    
    # learner

    # collector
    COLLECTOR_PUT_EXPS = 30
    COLLECTOR_GET_TRAINING_DATA = 31
    COLLECTOR_GET_BUFFER_LENGTH = 32

    # recorder
    STATS_RECORDER_PUT_INTERACT_SUMMARY = 40
    # policy_mgr
    POLICY_MGR_PUT_POLICY = 70
    POLICY_MGR_GET_POLICY = 71

class Msg(object):
    type: MsgType
    data: Optional[Any] = None