# 程序说明 05-Pro
## Folder Description

- `dataset`: 包含大部分故障诊断公开数据集的读取。为方便多个工程使用，getitem函数做了统一的规范

- `loss_fn`：自定义损失函数

- `networks`：
    - `classic_network`：包含复现论文的算法，目前与3个
	- `my_nets`：自己设计的深度残差滤波网络
	    - `DRFNs-enableRecord.py`：是验证可行的网络结构
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