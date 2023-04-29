import d4rl
import numpy as np
# in Offline RL, we use the env from d4rl and the dataset is in the env.
class Trainer:
    def __init__(self):
        self.buffer_empty = True

    def train_one_episode(self, env, agent, cfg):
        if self.buffer_empty:
            agent.buffer = d4rl.qlearning_dataset(env)
            self.buffer_empty = False
            agent.buffer_size = agent.buffer["observations"].shape[0]
            print("load the memories successfully!")
            print("buffer size: {}".format(agent.buffer["observations"].shape[0]))

        # train the agent
        agent.update(cfg.iters_per_ep)

        # evaluate the performance of the agent
        _, res = self.test_one_episode(env, agent, cfg)

        return agent, res

    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0
        ep_step = 0
        ep_frames = []
        state = env.reset(seed=cfg.seed)
        for _ in range(cfg.max_steps):
            ep_step += 1
            if cfg.render_mode == 'rgb_array':  # 用于可视化
                frame = env.render("rgb_array")
                ep_frames.append(frame)
            action = agent.predict_action(state)  # sample action
            next_state, reward, terminated, info = env.step(action)
            state = next_state  # update next state for env
            ep_reward += reward
            if terminated:
                break
        res = {'ep_reward': ep_reward, 'ep_step': ep_step, "ep_frames": np.array(ep_frames)}
        return agent, res