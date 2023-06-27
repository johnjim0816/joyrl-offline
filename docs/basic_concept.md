# 基本概念

该部分主要讲述`JoyRL`的基本原理与框架说明

## 基础回顾

在强化学习中，智能体（agent）在环境（environment）中与环境进行交互，不断更新自身的策略，以获得最大化的奖励（reward），如下图：

<div align=center>
<img width="400" src="../figs/interaction_mdp.png"/>
</div>

在这个过程中会有以下几个基本元素：

* 智能体：即策略与模型的载体，负责与环境进行交互，并不断更新自身的策略
* 环境：负责与智能体进行交互，并与智能体共同产生训练样本
* 经验池：负责存储智能体与环境交互的样本，样本就是我们所说的`transition`，包括`state`、`action`、`reward`、`next_state`等等

强化学习的基本训练逻辑，可以用如下伪代码表示：

```python
for i_episode in range(n_episodes): # 智能体与环境交互以回合(episode)为单位
    state = env.reset() # 重置环境
    for t in range(max_steps): # 每回合的长度
        action = agent.sample(state) # 智能体根据当前状态选择动作
        next_state, reward, done, _ = env.step(action) # 环境根据动作返回下一个状态、奖励、是否终止等信息
        agent.buffer.push(state, action, reward, next_state, done) # 将交互样本存入经验池
        agent.update() # 智能体更新策略
        state = next_state # 更新状态
        if done: # 如果终止，跳出当前回合
            break
```

## 数据流

从前面的训练逻辑中我们知道，强化学习的训练方式跟深度学习是有所不同的。深度学习中，我们需要将数据集一次性加载到内存中，然后进行训练。而强化学习中，我们需要不断地与环境进行交互，产生训练样本，然后将样本存入经验池，再从经验池中采样出一批样本，进行训练。对于简单的强化学习算法，数据流向比较简单，只有从环境存放到经验池，再从经验池随机采样到智能体中进行更新。但是对于复杂的算法，数据流向往往会更多。比如对于`PER`算法，我们需要把智能体在算法更新过程中产生的`TD`误差传回经验池以便于更新对应的`buffer`参数。再比如对于`PPO`算法，我们需要把智能体在选择动作时的`log_prob`保存下来，以便于在更新策略时当作`old policy`的一部分使用。

综合以上种种，`JoyRL`设计了一套统一的数据流向。首先将智能体、环境等几个基本元素进行拆分，主要包含以下几个部分：

* `Interactor`：即交互器，负责加载环境(`env`)以及从`Learner`那里获取策略(`policy`)并与环境交互，产生交互样本；
* `Learner`：即学习器，是`policy`的载体，负责加载训练样本并进行算法更新，同时产生策略样本；
* `Collector`：即收集器，是`buffer`的载体，负责接收交互器产生的交互样本和策略样本，并存入`buffer`中，同时负责从`buffer`中采样出训练样本；
* `Dataserver`：即数据服务器，负责一些全局变量，主要用于多进程的使用；

除此之外，还有一些其他的模块，比如记录奖励和损失曲线的`StatsRecorder`、在线测试模块`OnlineTester`等等，这些后面再详细介绍。基于以上几个模块，我们可以画出一个通用的数据流向图：

<div align=center>
<img width="500" src="../figs/data_flow.png"/>
</div>

图中有两个比较重要的数据流向，其一是`Exps(After interact)`，即交互样本，主要包括`state`、`action`、`reward`、`next_state`、`done`等，也包括`PPO`中的`log_prob`等信息，其二是`Exps(After learn)`，即策略样本，是策略更新后产生的一些信息，比如`TD`误差，`PPO`中的`ratio`等等，当然这些信息并不会在所有算法中用到，所以一般不会传入或者传入一个空值到`Collector`中。这些我们统称为样本或者经验，即`Exps`，很多情况下样本并不是直接进入到`buffer`中的，比如在`HER`之类的算法中，我们往往需要根据交互样本中的`state`、`reward`等定义一个`goal`用来帮助训练。策略样本也是，在`PER`中并不是直接把`TD`误差简单传入进去，而是要更新`buffer`中的`priority`，这就是为什么要定义`Collector`模块的原因。为了让用户能够自定义一些样本的处理方式，`JoyRL`中的`Collector`模块提供了一个`data_handler`接口，用户可以根据自己的需求自定义一些样本的处理方式。

## 训练器

结合基本的训练逻辑和以上定义的模块，我们可以用伪代码表示`JoyRL`中的训练过程(为了表述更清晰，所有模块都以大写表示)：

```python
while True:
    policy = LEARNER.get_policy() # 从learner中获取policy
    interact_outputs = [INTERACTOR.run(policy) for INTERACTOR in interactors] # 运行交互器并得到交互样本
    COLLECTOR.add_exps_list([interact_output['exps'] for interact_output in interact_outputs]) # 将经验添加到buffer中
    for _ in range(n_steps_per_learn): # 每次学习多少步
        training_data = COLLECTOR.get_training_data() # 从buffer中采样出训练样本
        learner_output = LEARNER.run(training_data) # 运行learner进行算法更新
        if learner_output is not None:
            policy = LEARNER.get_policy() # 从learner中获取更新后的policy
            COLLECTOR.handle_data_after_learn(learner_output['policy_data_after_learn']) # 处理策略样本
    if DATASERVER.check_task_end(): # 利用dataserver检查是否达到终止条件 
        break  
```
训练流程图如下所示：

<div align=center>
<img width="600" src="../figs/train_procedure.png"/>
</div>

注意，除了前面提到的几个模块之外，`JoyRL`还定义了一些其他的模块，比如`StatsRecorder`，用来记录交互过程中产生的奖励以及算法更新过程中产生的损失曲线等等，并写到`tensorboard`等文件中。而`OnlineTester`模块用来在线测试，即在训练过程中定期测试策略的性能，这个模块在`JoyRL`中是可选的，用户可以根据自己的需求选择是否使用。整个训练流程就是：

* 从`Learner`中获取`policy`，并在`Interactor`中运行，产生交互样本；
* 交互样本中的一部分数据比如`reward`等会被记录到`StatsRecorder`中，另一部分数据会被传入到`Collector`中，`Collector`会根据用户自定义的`data_handler`对样本进行处理，比如`HER`中的`goal`等；
* 从`Collector`中采样出训练样本，传入到`Learner`中进行算法更新；
* 更新之后产生的策略样本会被传入到`Collector`中，`Collector`会根据用户自定义的`data_handler`对样本进行处理，比如`PER`中的`priority`等；同时更新的部分数据比如`loss`等会被记录到`StatsRecorder`中，并且`policy`会定期传入到`OnlineTester`中进行在线测试；
* 重复以上过程直到达到终止条件。

这样的训练过程在`JoyRL`中定义为一个`Trainer`，即训练器，这是一个最基本的训练器。对于一些复杂的算法比如`offline`之类，会定义额外的训练器来训练，比如`OfflineTrainer`，这些训练器都是继承自`BaseTrainer`的，用户也可以根据自己的需求自定义一些训练器。


## 目录树

`JoyRL`的目录树如下所示：

```python
|-- joyrl
    |-- algos # 算法文件夹
        |-- [Algorithm name] # 指代算法名称比如DQN等
            |-- config.py # 存放每个算法的默认参数设置
                |-- class AlgoConfig # 算法参数设置的类
            |-- policy.py # 存放策略
                |-- class Agent # 每个算法的类命名为Agent
            |-- config.py
    |-- config 
        |-- config.py # 存放通用参数设置
    |-- framework # 框架文件夹，存放一些模块等
    |-- presets # 预设的参数，对应的结果存放在benchmarks下面
    |-- envs
    |-- benchmarks # 存放训练好的结果
    |-- docs # 说明文档目录
    |-- tasks # 训练的时候会自动生成
    |-- main.py # JoyRL训练主函数
    |-- README.md # 项目README
    |-- README_cn.md # 项目中文README
    |-- requirements.txt # Pyhton依赖列表
```