# 快速开始

## 强化学习回顾

在强化学习中，智能体（agent）在环境（environment）中与环境进行交互，不断更新自身的策略，以获得最大化的奖励（reward），如下图：

<div align=center>
<img width="400" src="../figs/interaction_mdp.png"/>
</div>

在交互过程中会有四种元素：

* 智能体：负责与环境进行交互(`agent.sample(state)`)，并更新自身的策略(`agent.update(state, action, reward, next_state, done)`)。
* 环境：负责与智能体进行交互(`env.step(action)`)，并返回下一个状态(`next_state`)、奖励(`reward`)、是否结束(`done`)等信息。
* 经验池：负责存储智能体与环境交互的样本(`agent.buffer.push(state, action, reward, next_state, done)`)，并在训练时从中采样(`agent.buffer.sample(batch_size)`)，在`JoyRL`离线版中，我们将经验池封装到了智能体中，而在`JoyRL`在线版中，我们将经验池用额外的模块封装起来，具体参考`框架说明`部分。
* 交互过程：即智能体与环境交互的过程。

交互过程一般可以用伪代码描述，也可以称之为强化学习训练接口，不同的强化学习算法，其训练接口也不尽相同，但大体是相似的，如下：

```python
for i_episode in range(n_episodes):
    state = env.reset()
    for t in range(max_steps):
        action = agent.sample(state)
        next_state, reward, done, _ = env.step(action)
        agent.buffer.push(state, action, reward, next_state, done)
        agent.update()
        state = next_state
        if done:
            break
```
[joyrl-book](https://github.com/datawhalechina/joyrl-book/tree/main/pseudocodes)提供了丰富的强化学习算法伪代码，帮助读者们更好地理解算法，也欢迎多多`star`～。在`JoyRL`中，我们将训练接口封装到了`Trainer`中，而在多线程中，我们将训练接口封装到了`Worker`中，具体参考相关说明部分。

## 创建环境

`JoyRL`中的环境主要有两种，一种是`gym`环境，一种是自定义环境，两者皆遵循`gym`接口。

```python
import gym
env = gym.make('CartPole-v1')
``` 
## 创建智能体

对于`DRL`算法，`JoyRL`中的智能体主要包含两个元素，一个是网络，一个是经验池。

首先定义网络，例如在`DQN`算法中我们可以定义一个简单的全连接网络，如下：

```python
class Qnet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """ 初始化q网络，为全连接网络
            state_dim: 环境的状态维度
            action_dim: 输出的动作维度
        """
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim) # 输入层
        self.fc2 = nn.Linear(hidden_dim,hidden_dim) # 隐藏层
        self.fc3 = nn.Linear(hidden_dim, action_dim) # 输出层
        
    def forward(self, x):
        # 各层对应的激活函数
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)
```

然后定义经验池，如下：

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity # 经验回放的容量
        self.buffer = [] # 缓冲区
        self.position = 0 
    
    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity 
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size) # 随机采出小批量转移
        state, action, reward, next_state, done =  zip(*batch) # 解压成状态，动作等
        return state, action, reward, next_state, done
    
    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)
```

最后定义智能体，如下：

```python
class Agent:
    def __init__(cfg):
        self.qnet = Qnet(cfg.state_dim, cfg.action_dim) # 定义q网络
        self.target_qnet = Qnet(cfg.state_dim, cfg.action_dim) # 定义目标q网络
        self.target_qnet.load_state_dict(self.qnet.state_dict()) # 将q网络的参数复制到目标q网络
        self.buffer = ReplayBuffer(cfg.capacity) # 定义经验池
    def sample(self, state):
        ''' 根据状态采样动作
        '''
        pass
    def predict(self, state):
        ''' 根据状态预测动作
        '''
        pass
    def update(self):
        ''' 更新网络
        '''
```
这里省去了一些细节，具体可以参考`JoyRL`中的`DQN`算法。

## 创建训练器

如下是`DQN`算法的训练器：

```python
class Trainer:
    def __init__(self) -> None:
        pass
    def train_one_episode(self, env, agent, cfg): 
        ep_reward = 0  # 每回合的reward之和
        ep_step = 0 # 每回合的step之和
        state = env.reset(seed = cfg.seed)  # 重置环境并返回初始状态
        for _ in range(cfg.max_steps):
            ep_step += 1
            action = agent.sample_action(state)  # 采样动作
            next_state, reward, terminated, truncated , info = env.step(action)  # 更新环境并返回转移
            agent.memory.push(state, action, reward, next_state, terminated)  # 存储样本(转移)
            agent.update()  # 更新智能体
            state = next_state  # 更新下一个状态
            ep_reward += reward   
            if terminated:
                break
        res = {'ep_reward':ep_reward,'ep_step':ep_step}
        return agent,res
    def test_one_episode(self, env, agent, cfg):
        ep_reward = 0  
        ep_step = 0
        ep_frames = []
        state = env.reset(seed = cfg.seed)  
        for _ in range(cfg.max_steps):
            ep_step += 1
            if cfg.render and cfg.render_mode == 'rgb_array': # 用于可视化
                frame = env.render()[0]
                ep_frames.append(frame)
            action = agent.predict_action(state) # 预测动作
            next_state, reward, terminated, truncated , info = env.step(action)  
            state = next_state  
            ep_reward += reward  
            if terminated:
                break
        res = {'ep_reward':ep_reward,'ep_step':ep_step,'ep_frames':ep_frames}
        return agent,res
```
这里`JoyRL`提供了`train_one_episode`和`test_one_episode`两个函数，分别用于训练和测试。

## 超参数设置

在`JoyRL`中，我们可以通过`yaml`文件来设置超参数，如下：

```yaml
general_cfg:
  algo_name: DQN
  device: cpu
  env_name: CartPole-v1
  mode: train
  load_checkpoint: false
  load_path: Train_CartPole-v1_DQN_20221026-054757
  max_steps: 200
  save_fig: true
  seed: 1
  show_fig: false
  test_eps: 10
  train_eps: 100
algo_cfg:
  value_layers:
    - layer_type: linear
      layer_dim: ['n_states',256]
      activation: relu
    - layer_type: linear
      layer_dim: [256,256]
      activation: relu
    - layer_type: linear
      layer_dim: [256,'n_actions']
      activation: none
  batch_size: 64
  buffer_size: 100000
  epsilon_decay: 500
  epsilon_end: 0.01
  epsilon_start: 0.95
  gamma: 0.95
  lr: 0.0001
  target_update: 4
```
其中包括通用参数`general_cfg`和算法参数`algo_cfg`，也可以通过`python`文件来设置超参数，通用的默认参数在`config/config.py`中，算法参数则在各自算法文件夹下的`config.py`中，这些`python`文件均对各参数做了注释说明。此外，当两者都设置时，`yaml`文件中的参数会覆盖`python`文件中的参数。

## 训练模式

配置好参数之后，可以直接执行```python main.py --yaml [yaml_file]```来训练。

## 测试模式

将通用参数中的`mode: train`改为`mode: test`，加载模型时将`load_checkpoint`改成`True`，`load_path`改成`tasks`路径下已经训好的文件夹，然后执行```python main.py --yaml [yaml_file]```即可。
## 收集模型

对于`BC`等算法，需要收集数据，可以将通用参数中的`mode: train`改为`mode: collect`，然后执行```python main.py --yaml [yaml_file]```即可。
## 渲染模式

渲染模式主要有两个参数来控制：
```yaml
render: True # 是否渲染
render_mode: human # 渲染模式
```
当渲染模式为`human`时，会在屏幕上渲染。

当渲染模式为`rgb_array`时，会返回一个`numpy`数组，在`JoyRL`中会在`[task_dir]/videos`下生成`video.gif`文件。

另外注意，`JoyRL`中的只有单线程下支持渲染模式。
