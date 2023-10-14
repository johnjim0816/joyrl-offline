from enum import Enum

class EnvMsgType(Enum):
    RESET = 0
    STEP = 1
    CLOSE = 2