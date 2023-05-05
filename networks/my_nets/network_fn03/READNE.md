
# 程序说明 @[network_fn03]
本程序`network_fn03`是延续`network_fn02`的版本。


## Folder Description

- `mynet03.py`：是用于训练的网络结构 || 用于调试网络参数与结构
- `train_mynet03.py`：针对该网络的训练函数，主要解决网络的个性化输入/输出问题
- `run_mynet03.py`：个性化开始训练的函数，以及评估网络、可视化显示结果。
- `DRFNs-enableRecord.py`：保存训练效果较好的网络结构。

***

## Summary of `Network_fn03`
- 采用注意力机制，自动计算DRFN网络中的滤波器尺寸；
- 每个滤波单元也在最后使用Maxpool减小（1/2）特征图；
- 可选的，通过卷积网络ConV1d，再次减小（1/2）的特征图。

## Research Finding

- 滤波器尺寸自动计算，最终每个滤波单元的滤波器尺寸不一定相同；
- 整体的故障诊断精度有所下降，最高约97%；
- 在新数据集（新场景/新工况）上，每个单元的滤波器尺寸是一致的；
- 这说明，注意力机制在跨工况/跨设备的故障诊断上具有较大潜力。

> 为此，我们开展`network_fn04`的研究


[network_fn03]: /networks/my_nets/network_fn03/README.md