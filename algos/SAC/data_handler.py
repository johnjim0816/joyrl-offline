import numpy as np
from algos.base.data_handlers import BaseDataHandler
class DataHandler(BaseDataHandler):
    def __init__(self, cfg):
        super().__init__(cfg)
    def handle_exps_before_train(self, exps):
        ''' convert exps to training data
        '''
        states = np.array([exp.state for exp in exps])
        # print(exps[-1].__dict__)
        # print(f"exps[-1].action type:{type(exps[-1].action)}")
        # print(f"exps[-1].action:{exps[-1].action}")
        # print(f"exps[-1].states:{exps[-1].state}")
        if type(exps[-1].action) is np.ndarray:
            actions = np.array([exp.action for exp in exps])
        else:
            actions = np.array([[exp.action] for exp in exps])
            # print(f"actions value :{actions}")
        rewards = np.array([[exp.reward] for exp in exps])
        next_states = np.array([exp.next_state for exp in exps])
        dones = np.array([[exp.done] for exp in exps])
        # discrete
        # print(f"exp.probs:{exps[-1].probs}")
        # xxx
        probs = [exp.probs for exp in exps] if hasattr(exps[-1],'probs') else None
        log_probs = [exp.log_probs for exp in exps] if hasattr(exps[-1],'log_probs') else None
        # continue
        # print(f"exp.mu:{exps[-1].mu}")
        # xxx
        value = [exp.value[0] for exp in exps] if hasattr(exps[-1],'value') else None
        mu = [exp.mu[0] for exp in exps] if hasattr(exps[-1],'mu') else None
        sigma = [exp.sigma[0] for exp in exps] if hasattr(exps[-1],'sigma') else None
        data = {'states': states, 'actions': actions, 'rewards': rewards, 'next_states': next_states, 'dones': dones, 
                'probs': probs, 'log_probs': log_probs, 'value': value, 'mu': mu, 'sigma': sigma}
        # print(f"data:{data}")
        # xx
        return data