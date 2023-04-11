class Trainer:
    '''训练
    '''
    def __init__(self) -> None:
        pass
    def train_one_episode(self, env, agent, cfg): 
        ep_reward = 0 # 每一回合的奖励，初始化为 0                    
        ep_step = 0 # 每一回合的步数
        state = env.reset() # 重置环境并且获取初始状态                        
        for _ in range(cfg.max_steps):
            ep_step += 1 # 累加回合步数
            action = agent.sample_action(state)  # 抽样动作                       
            next_state, reward, terminated, truncated , info = env.step(action)  # 更新环境，返回 transitions   
            agent.memory.push(state, action, reward,next_state, terminated)  # 保存 transitions
            agent.update()  # 更新智能体                 
            state = next_state  # 更新状态             
            ep_reward += reward  # 累加奖励
            if terminated:
                break
        return agent,ep_reward,ep_step
    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  # 每一回合奖励，初始化为 0      
        ep_step = 0
        state = env.reset()  # 重置环境并且获取初始状态           
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.predict_action(state)  # 抽样动作                 
            next_state, reward, terminated, truncated , info = env.step(action)  # 更新环境，返回 trasitions   
            state = next_state  # 更新状态           
            ep_reward += reward  # 累加奖励
            if terminated:
                break
        return agent,ep_reward,ep_step