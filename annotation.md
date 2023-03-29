# JoyRL 注释规范

## 0. 通用Python规范

参考[Google 开源项目风格指南](https://zh-google-styleguide.readthedocs.io/en/latest/google-python-styleguide/python_style_rules/)，着重牢记以下几个要点：

* 每行不超过80个字符
* 不要滥用括号

单行注释以 `#` 开头，多行注释用三个单引号 `'''` 包裹（在JoyRL中不要用双引号）

## 1. 中文注释规范

### 1.1. 中文注释原则

中文注释应该尽量简洁，不要写太多无用的信息，只写必要的信息。注释应该是对代码的补充，而不是代码的替代。

### 1.2. 中文注释细则

1. 中文注释使用中文标点符号，不要使用英文标点符号。

2. 中文注释中包含英文时，应在中英文之间保持空格。
3. 所有类、函数、方法、变量的注释都应该写在其定义处，而不是使用处，用多行注释。
如下，这里
```python
class Agent:
    def __init__(self,cfg, is_share_agent = False):
        '''智能体类
        Args:
            cfg (class): 超参数类
            is_share_agent (bool, optional): 是否为共享的 Agent ，多进程下使用，默认为 False
        '''
agent = Agent(cfg) # 这里使用处不用写注释
```
4. 类或函数的注释应该应在定义中包含功能，输入参数，返回值等信息，如下：
```python
def sample_action(self, state):
     ''' 采样动作
     Args:
         state (array): 状态
     Returns:
         action (int): 动作
     '''
```
这里推荐使用[autoDocstring](https://marketplace.visualstudio.com/items?itemName=njpwerner.autodocstring)插件，可以自动生成注释模板，使用默认的谷歌风格即可。**注意，这个插件自动生成的注释会有多余的空行，按照上面的规范删除即可**。**另外强烈建议用Github Copilot来配合写注释，会方便很多。**

5. 所有的超参数类下的参数都应该逐行注释，如下

```python
class AlgoConfig(DefaultConfig):
    '''算法超参数类
    '''
    def __init__(self) -> None:
        ## 设置 epsilon_start=epsilon_end 可以得到固定的 epsilon，即等于epsilon_end
        self.epsilon_start = 0.95  # epsilon 初始值
        self.epsilon_end = 0.01  # epsilon 终止值
        self.epsilon_decay = 500  # epsilon 衰减率
        self.gamma = 0.95  # 奖励折扣因子
        self.lr = 0.0001  # 学习率
        self.buffer_size = 100000  # buffer 大小
        self.batch_size = 64  # batch size
        self.target_update = 4  # target_net 更新频率
        self.value_layers = [
            {'layer_type': 'linear', 'layer_dim': ['n_states', 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 256],
             'activation': 'relu'},
            {'layer_type': 'linear', 'layer_dim': [256, 'n_actions'],
             'activation': 'none'}] # 神经网络层配置
```
但这些超参数在智能体类下的注释中不需要再写一遍，遵循第3条原则，如下：
```python
class Agent:
    def __init__(self,cfg, is_share_agent = False):
        '''智能体类
        Args:
            cfg (class): 超参数类
            is_share_agent (bool, optional): 是否为共享的 Agent ，多进程下使用，默认为 False
        '''
        self.n_actions = cfg.n_actions  
        self.device = torch.device(cfg.device) 
        self.gamma = cfg.gamma  
        ## e-greedy 策略相关参数
        self.sample_count = 0  # 采样动作计数
        self.epsilon_start = cfg.epsilon_start
        self.epsilon_end = cfg.epsilon_end
        self.epsilon_decay = cfg.epsilon_decay
        self.batch_size = cfg.batch_size
        self.target_update = cfg.target_update
```
6. 网络模型中定义线性层等不必注释，定义复杂的网络层必须注释
7. 给某段代码注释，需在代码前一行注释，并用`##`开始，**这是JoyRL特有的规则**。
8. 算法更新部分即update函数必须逐行注释，这个是最重要的地方

## 2. 英文注释规范

除了标点符号使用英文标点，不要使用中文标点，其他均与中文注释类似。