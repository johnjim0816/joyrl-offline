import yaml
from copy import deepcopy as dcp
import pickle
from pathlib import Path
import os

class Trainer:
    
    def __init__(self):
        self.behave_agent = None
        self.buffer_empty = True

    def train_one_episode(self,env,agent,cfg):
        if self.buffer_empty:
            # load buffer
            current_path = os.path.abspath(os.path.dirname(__file__))
            traj_pkl = current_path+'/traj/traj.pkl'

            # load the buffer
            with open(traj_pkl,"rb") as f:
                traj = dcp(pickle.load(f))
                #print(traj["states"][0])
                for trans_i in range(len(traj['states'])):
                    agent.buffer.push(state=traj["states"][trans_i],
                                      action=traj["actions"][trans_i],
                                      reward=traj["rewards"][trans_i],
                                      next_state=traj["next_states"][trans_i],
                                      done=traj["terminals"][trans_i])
            self.buffer_empty = False
            print("load the memories successfully!")
        # train
        agent.train(cfg.iters_per_ep)
        # evaluate 
        _, res = self.test_one_episode(env,agent,cfg)
        
        return agent, res
        
    def test_one_episode(self, env, agent, cfg):
        
        ep_reward = 0  # reward per episode
        ep_step = 0
        state = env.reset()  # reset and obtain initial state
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.select_action(state)  # sample action
            next_state, reward, terminated, truncated, info = env.step(
                action)  # update env and return transitions under new_step_api of OpenAI Gym
            state = next_state  # update next state for env
            ep_reward += reward
            if terminated:
                break
        res = {'ep_reward': ep_reward, 'ep_step': ep_step}
        return agent, res
    
    def collect_one_episode(self,env, agent, cfg):
        if self.behave_agent == None:
            # obtain the behavior agent
            behave_algo_mod = __import__(f"algos.{cfg.behavior_agent_name}.config",
                       fromlist=['AlgoConfig'])
            behave_cfg = behave_algo_mod.AlgoConfig()
            behave_cfg.n_states = cfg.n_states
            behave_cfg.n_actions = cfg.n_actions
            behave_cfg.action_space = cfg.action_space

            current_path = os.path.abspath(os.path.dirname(__file__))
            #print(current_path)
            with open(current_path+cfg.behavior_agent_parameters_path) as f:
                behave_cfg_yml = yaml.load(f, Loader=yaml.FullLoader)
                for k,v in behave_cfg_yml.items():
                    setattr(behave_cfg,k,v)

            algo_mod = __import__(f"algos.{cfg.behavior_agent_name}.agent", fromlist=['Agent'])
            #print("algo mode: ",algo_mod)
            behave_agent = algo_mod.Agent(behave_cfg)

            # load the actor model for behavioural agent
            behave_agent.load_model(current_path+cfg.behavior_policy_path)
            print("load the behaviorial agent model successfully")

            self.behave_agent = behave_agent
            self.behave_cfg = behave_cfg
        else:
            behave_agent = self.behave_agent

        total_reward = 0
        ep_state, ep_action, ep_next_state, ep_reward, ep_terminal = [], [], [], [], []
        
        step = 0 
        state = env.reset()
        for _ in range(self.behave_cfg.max_steps):
            step+=1
            if cfg.collect_explore_data:
                # explore:
                action = behave_agent.sample_action(state)
            else:
                # deterministic action:
                action = behave_agent.predict_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            
            total_reward+=reward
            ep_state.append(dcp(state))
            ep_action.append(dcp(action))
            ep_next_state.append(dcp(next_state))
            ep_reward.append(reward)
            ep_terminal.append(terminated)

            state = next_state

            if terminated:
                break
        
        return total_reward, ep_state, ep_action, ep_next_state, ep_reward, ep_terminal
    