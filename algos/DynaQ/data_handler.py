import numpy as np
from algos.base.data_handlers import BaseDataHandler
from algos.base.exps import Exp


class DataHandler(BaseDataHandler):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.buffer = []
        self.data_after_train = {}

    def add_transition(self, transition):
        ''' add transition to buffer
        '''
        exp = self._create_exp(transition)
        self.buffer.append(exp)

    def add_data_after_train(self, data):
        ''' add update data
        '''
        self.data_after_train = data

    def sample_training_data(self):
        ''' sample training data from buffer
        '''
        exp = self.buffer.pop()[0]
        if exp is not None:
            return self.handle_exps_before_train(exp)
        else:
            return None

    def _create_exp(self, transtion):
        ''' create experience
        '''
        return [Exp(**transtion)]

    def handle_exps_before_train(self, exp, **kwargs):
        ''' convert exps to training data
        '''
        state = np.array(exp.state)
        action = np.array(exp.action)
        reward = np.array(exp.reward)
        next_state = np.array(exp.next_state)
        done = np.array(exp.done)
        data = {'state': state, 'action': action, 'reward': reward,
                'next_state': next_state, 'done': done}
        return data

    def handle_exps_after_train(self):
        ''' handle exps after train
        '''
        pass
