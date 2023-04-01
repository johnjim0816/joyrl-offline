# 如何贡献

欢迎广大开发爱好者为 JoyRL 贡献代码，如果你想参与bug修复的话，直接修改对应的代码然后PR即可，PR教程可参考[VS Code快速实现Git PR操作](https://blog.csdn.net/JohnJim0/article/details/128156442)。如果你想贡献新的算法的话，可以按照以下步骤进行，有问题的话随时交流～（微信：johnjim0816）

## 新建算法

首先在`algos`目录下新建文件夹，明明为你想要新增的算法名称，并且在`config.py`下配置好默认参数

## 配置参数

在`presets`下配置好`yaml`文件，包括`Train`和`Test`的部分

## 运行代码

调试好你的算法代码之后，分别训练和测试一次，将对应输出的文件夹放到`benchmarks`目录下

## 代码注释

[注释说明](./annotation.md), 本仓库统一中文注释



## 修改文档

在`docs/hyper_tuning.md`文件中写好你贡献的算法的参数说明
## 著权署名

在`README.md`的算法列表中增加一行，写上对应的算法和自己的名字，最后PR即可