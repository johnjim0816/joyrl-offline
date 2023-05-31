import random
import numpy as np

def calcu_reward(new_goal, state, action):
    # direcly use observation as goal
    goal_cos, goal_sin, goal_thdot = new_goal[0], new_goal[1], new_goal[2]
    cos_th, sin_th, thdot = state[0], state[1], state[2]
    costs = angle_normalize(np.arccos(goal_cos) - np.arccos(cos_th)) ** 2 + 0.1 * (goal_thdot - thdot) ** 2#+ (goal_sin - sin_th) ** 2 
    reward = -costs
    return reward

def generate_goals(i, episode_cache, sample_num, sample_range = 200):
    '''
    Input: current steps, current episode transition's cache, sample number
    Return: new goals sets
    notice here only "future" sample policy
    '''
    end = (i+sample_range) if i+sample_range < len(episode_cache) else len(episode_cache)
    epi_to_go = episode_cache[i:end]
    if len(epi_to_go) < sample_num:
        sample_trans = epi_to_go
    else:
        sample_trans = random.sample(epi_to_go, sample_num)
    return [np.array(trans[3][:3]) for trans in sample_trans]#episode_cache.append((o,a,r,o2))

def angle_normalize(x):
    return ((x + np.pi) % (2 * np.pi)) - np.pi


class Trainer:
    def __init__(self) -> None:
        pass

    def train_one_episode(self, env, agent, cfg):
        '''
        更新一轮参数
        Args:
            env (gym): 输入的env实例
            agent (class): 输入的agent实例
            cfg (class): 超参数配置实例

        Returns:
            agent (class): 更新一轮参数后的agent实例
            res (dict): 更新一轮后的总奖励值及更新的step总数
        '''
        ep_reward = 0  # reward per episode
        ep_step = 0
        episode_cache = []
        HER_SAMPLE_NUM = cfg.her_sample_num
        update_every = cfg.update_every
        state = env.reset()  # 环境状态
        # 随机构造目标
        np.random.seed(cfg.seed)
        costheta = np.random.rand()
        sintheta = np.sqrt(1-costheta**2)
        w = 2 * np.random.rand()
        goal = np.array([costheta,sintheta,w])
        state = np.concatenate((state, goal)) # 结合环境状态和目标，构造新状态
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # sample action
            next_state, reward, terminated, truncated, info = env.step(
                action)  # update env and return transitions under new_step_api of OpenAI Gym
            next_state = np.concatenate((next_state, goal)) # 结合下一时间步环境状态和目标，构造新下一时间步状态
            reward = calcu_reward(goal, state, action) # 根据目标、状态、动作计算HER奖励
            episode_cache.append((state, action, reward, next_state)) # 缓存transitions，便于事后经验回放
            agent.memory.push((state, action, reward,
                               next_state, terminated))  # 将transitions存入经验缓存池
            
            state = next_state  # update next state for env
            ep_reward += reward  #
            if terminated:
                break

        # Hindsight replay: Important operation of HER
        for i, transition in enumerate(episode_cache):
            new_goals = generate_goals(i, episode_cache, HER_SAMPLE_NUM) # 根据future方法构造新目标
            for new_goal in new_goals:
                state, action = transition[0], transition[1] # 从transition提取具有目标的状态和动作
                reward = calcu_reward(new_goal, state, action) # 根据新目标、状态、动作计算奖励值
            
                state, new_state = transition[0][:3], transition[3][:3] # 从transition提取不包含目标的状态和下一时间步状态
                state = np.concatenate((state, new_goal)) # 结合环境状态和生成的新目标，构造新状态
                new_state = np.concatenate((new_state, new_goal)) # 结合下一时间步环境状态和生成的新目标，构造新下一时间步状态
                agent.memory.push((state, action, reward, new_state, False)) # 将新的transition存入经验缓存池


        for _ in range(update_every):
            agent.update()  # update agent

        res = {'ep_reward': ep_reward, 'ep_step': ep_step}
        return agent, res



    def test_one_episode(self, env, agent, cfg):
        '''
        预测一轮
        Args:
            env (gym): 输入的env实例
            agent (class): 输入的agent实例
            cfg (class): 超参数配置实例

        Returns:
            agent (class): 执行完一轮后的agent实例
            res (dict): 执行一轮后的总奖励值及预测的step总数
        '''
        ep_reward = 0  # reward per episode
        ep_step = 0
        ep_frames = []
        state = env.reset(seed = cfg.seed)  # 环境状态
        # 随机构造目标
        costheta = np.random.rand()
        sintheta = np.sqrt(1-costheta**2)
        w = 2 * np.random.rand()
        goal = np.array([costheta,sintheta,w])
        state = np.concatenate((state, goal)) # 结合环境状态和目标，构造新状态
        for _ in range(cfg.max_steps):
            ep_step += 1
            if cfg.render and cfg.render_mode == 'rgb_array': # 用于可视化
                frame = env.render()[0]
                ep_frames.append(frame)
            action = agent.predict_action(state)  # sample action
            next_state, reward, terminated, truncated, info = env.step(
                action)  # update env and return transitions under new_step_api of OpenAI Gym
            next_state = np.concatenate((next_state, goal)) # 结合下一时间步环境状态和目标，构造新下一时间步状态
            state = next_state  # update next state for env
            ep_reward += reward
            if terminated:
                break
        res = {'ep_reward':ep_reward,'ep_step':ep_step,'ep_frames':ep_frames}
        return agent, res
