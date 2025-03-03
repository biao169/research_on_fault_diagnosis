
# 程序说明 @[network_fn02]
本程序`network_fn02`是调试的第一个版本，也是故障诊断系列网络的第一个版本。
# 引用
>[1] J. Tan, J. Wan, B. Chen, M. Safran, S. A. AlQahtani, and R. Zhang, “Selective Feature Reinforcement Network for Robust Remote Fault Diagnosis of Wind Turbine Bearing Under Non-Ideal Sensor Data,” IEEE Transactions on Instrumentation and Measurement, vol. 73, pp. 1-11, 2024, Art no. 3515911, doi: 10.1109/TIM.2024.3375958.
>
———————————————                        
译文链接：https://blog.csdn.net/tjb132/article/details/145962010


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
