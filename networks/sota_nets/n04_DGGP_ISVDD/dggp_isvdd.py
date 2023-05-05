# -*- coding:utf-8 -*-
__AUTHOR__ = 'Jinbiao Tan'
"""
  X. Ni, D. Yang, H. Zhang, F. Qu and J. Qin, "Time-Series Transfer Learning: An Early Stage Imbalance Fault Detection Method Based on Feature Enhancement and Improved Support Vector Data Description," in IEEE Transactions on Industrial Electronics, vol. 70, no. 8, pp. 8488-8498, Aug. 2023, doi: 10.1109/TIE.2022.3229351.
  
"""

import torch, os
from torch import nn


class LSTM_Normal(nn.Module):
    """ 构建 LSTM网络，建立隐藏层缓存，方便后期使用 """

    def __init__(self, in_channel=1, input_len=1024, output_len=5, num_layers=1, is_cache=True):
        super(LSTM_Normal, self).__init__()
        batch = in_channel
        hidden_len = output_len
        self.is_cache = None
        if is_cache: self.is_cache = [num_layers, batch, hidden_len]
        self.hidden = None  # (torch.rand([num_layers, batch, hidden_len]), torch.rand([num_layers, batch, hidden_len]))
        # if torch.cuda.is_available(): self.hidden = self.hidden.cu
        self.lstm = nn.LSTM(input_size=input_len, hidden_size=hidden_len, num_layers=num_layers)

    def forward(self, x):
        if self.is_cache is None: out, hidden = self.lstm(x, )
        else:
            if self.hidden is None:
                # self.is_cache[1] = len(x)
                self.hidden = (torch.rand(self.is_cache, device='cuda'), torch.rand(self.is_cache, device='cuda'))
            out, hidden = self.lstm(x,  self.hidden)
            self.hidden = (hidden[0].clone().detach(),hidden[1].clone().detach())
        # torch.tensor().clone()
        return out


""" ============================ 生成器 ========================================="""


class Generate_E(nn.Module):
    """ Generate_E： 输入严重故障前的数据，输出（推测）其之后的严重故障数据 """

    def __init__(self, in_channel=1, out_len=4, ):
        super(Generate_E, self).__init__()
        self.log = 'Generate_E'
        self.___file__ = os.path.abspath(__file__)

        # ---- input size: 2048 ----------

        self.generate_e = nn.Sequential(
            LSTM_Normal(in_channel=1, input_len=2048, output_len=2048, num_layers=1, is_cache=True),  # 2048-->1024

        )
        pass

    def forward(self, x):
        x = self.generate_e(x)
        return x


class Generate_S(nn.Module):
    """ Generate_S： 输入严重故障的数据，输出（推测）其之前的非严重故障数据 """

    def __init__(self, in_channel=1, out_len=4, ):
        super(Generate_S, self).__init__()
        self.log = 'Generate_S'
        self.___file__ = os.path.abspath(__file__)

        # ---- input size: 2048 ----------

        self.generate_e = nn.Sequential(
            LSTM_Normal(in_channel=1, input_len=2048, output_len=2048, num_layers=1, is_cache=True),  # 2048-->1024

        )
        pass

    def forward(self, x):
        x = self.generate_e(x)
        return x


""" 
    ============================ 鉴别器 =========================================
    (DGGP): 为双重生成梯度惩罚，采用双对抗网络进行自优化。主要用于生成故障数据，实现域迁移和自适应
    （Improved SVDD：ISVDD）：是故障诊断模型
"""


class Discriminator_E(nn.Module):
    """ Discriminator_E： 输入严重故障的数据，输出（推测）其之前的非严重故障数据 """

    def __init__(self, in_channel=1, out_len=4, ):
        super(Discriminator_E, self).__init__()
        self.log = 'Discriminator_E'
        self.___file__ = os.path.abspath(__file__)

        # ---- input size: 2048 ----------

        self.generate_e = nn.Sequential(
            LSTM_Normal(in_channel=1, input_len=2048, output_len=2048, num_layers=1, is_cache=True),  # 2048-->1024

        )
        pass

    def forward(self, x):
        x = self.generate_e(x)
        return x


class Discriminator_S(nn.Module):
    """ Discriminator_S： 输入严重故障的数据，输出（推测）其之前的非严重故障数据 """

    def __init__(self, in_channel=1, out_len=2048, ):
        super(Discriminator_S, self).__init__()
        self.log = 'Discriminator_S'
        self.___file__ = os.path.abspath(__file__)

        # ---- input size: 2048 ----------

        self.generate_e = nn.Sequential(
            LSTM_Normal(in_channel=1, input_len=2048, output_len=2048, num_layers=1, is_cache=True),  # 2048-->1024

        )
        pass

    def forward(self, x):
        x = self.generate_e(x)
        return x




def debug_net():
    """ 用于调试网络构建代码是否调通 """
    x = torch.rand([3, 1, 2048])
    net = WPD_MSCNN(1, 4)
    print('testing:', net.___file__)
    y = net(x)
    print(y)
    pass
