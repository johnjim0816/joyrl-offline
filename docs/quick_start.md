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


## 渲染模式

渲染模式主要有两个参数来控制：
```yaml
render: True # 是否渲染
render_mode: human # 渲染模式
```
当渲染模式为`human`时，会在屏幕上渲染。

当渲染模式为`rgb_array`时，会返回一个`numpy`数组，可以用`plt.imshow`来渲染，在`JoyRL`中会在`[task_dir]/videos`下生成`video.gif`文件。
