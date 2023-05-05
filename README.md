# 程序说明 04-Pro
## Folder Description

- `dataset`: 包含大部分故障诊断公开数据集的读取。为方便多个工程使用，getitem函数做了统一的规范

- `loss_fn`：自定义损失函数

- `networks`：
    - `classic_network`：包含复现论文的算法，目前与3个
	- `my_nets/network_fn02`：自己设计的深度残差滤波网络
	    - `DRFNs-enableRecord.py`：是论文提及的网络结构和参数
	    - `mynet02.py`： 是网络调试过程的网络结构搭建
- `report_make`：将模型的测试结果，打印成Excel表格
- `utils_tool`：包含日常用的工具 tensorboard
	- `log_utils.py`：训练日志（acc， loss）数据的保存。无需在for前定义全局函数，自动整理。
			文件内还有混稀矩阵绘制程序
	- `t_SNE.py`：分类特性效果图（散点图）

#### 每个 `network` 对应一个文件夹，内部包含网络结构的设计、训练方式的定义和训练-测试的主函数。

***
#### 05-Pro：DRFNs的滤波器核尺寸通过自注意力机制，自动确定。实验数据在：
	I:\python\01-work\result\05-pro-auto-kernelSize


# 程序说明 @[network_fn02]
本程序`network_fn02`是调试的第一个版本，也是故障诊断系列网络的第一个版本。

## Spark of Thought
- 受论文DRSN网络启发，深度学习网络在特征提炼过程中，应该会有针对性的学习一下特征；
- 我们认为每个特征向量中，值偏小的特征对最终结果贡献不大，属于干扰成分；
- 因此，考虑针对特征向量中的干扰成分进行剔除，同时增强有效成分的域；
- 卷积操作是一种可以将邻域信息填充到当前位置的算法，为此，考虑引入处理特征向量，后文也成为“滤波器”；


## Folder Description

- `mynet02.py`：是用于训练的网络结构 || 用于调试网络参数与结构
- `train_mynet02.py`：针对该网络的训练函数，主要解决网络的个性化输入/输出问题
- `run_mynet02.py`：个性化开始训练的函数，以及评估网络、可视化显示结果。
- `DRFNs-enableRecord.py`：保存训练效果较好的网络结构。

***

## Summary of `Network_fn02`
- DRFN网络中的滤波器尺寸采用人工设定；
- 滤波的方式，即滤波核参数（均值滤波、高斯滤波、自适应）也采用人工设定；
- 每个滤波单元最后默认不缩小特征图；
- 可选的，通过卷积网络ConV1d和Maxpool，减小（1/4）的特征图。

## Research Finding

- 噪声干扰下，故障诊断性能稳定性大幅上升；
- 整体的故障诊断精度基本稳定，最高约100%；

由于滤波器尺寸确定进行了大量调试实验，为此，我们希望这个尺寸可以自动确定。

> 为此，我们开展`network_fn03`的研究


[network_fn02]: /networks/my_nets/network_fn02/README.md
