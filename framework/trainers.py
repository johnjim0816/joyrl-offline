class BaseTrainer:
    def __init__(self,*args,**kwargs) -> None:
        self.interactors = kwargs['interactors']
        self.learners = kwargs['learners']
        self.collector = kwargs['collector']
        self.n_steps_per_learn = kwargs['n_steps_per_learn']
    def run(self):
        raise NotImplementedError
    
class SimpleTrainer(BaseTrainer):
    def __init__(self,*args,**kwargs) -> None:
        pass
    def run(self):
        while True:
            policy = self.learners[0].get_policy() # get policy from main learner
            interact_outputs = [interactor.run(policy) for interactor in self.interactors] # run interactors
            training_data = self.collector.handle_exps_after_interact(interact_outputs) # handle exps after interact
            for _ in range(self.n_steps_per_learn):
                learner_outputs = [learner.learn(training_data) for learner in self.learners] # learn
                self.collector.handle_exps_after_update(learner_outputs) # handle exps after update
            

class RayTrainer(BaseTrainer):
    def __init__(self,*args,**kwargs) -> None:
        pass