# -*- coding:utf-8 -*-
__AUTHOR__ = 'Jinbiao Tan'
"""
   D. Huang, W. -A. Zhang, F. Guo, W. Liu and X. Shi, "Wavelet Packet Decomposition-Based Multiscale CNN for Fault Diagnosis of Wind Turbine Gearbox," in IEEE Transactions on Cybernetics, vol. 53, no. 1, pp. 443-453, Jan. 2023, doi: 10.1109/TCYB.2021.3123667.
   风电装备故障诊断，应对温度、气压等多种干扰，设计一种稳定可靠的网络。
"""


import torch, os
from torch import nn
import pywt


class WPD_Pre_Data(nn.Module):
    """ wavelet packet decomposition """

    def __init__(self,):
        super(WPD_Pre_Data, self).__init__()
        self.log = 'WPD_Preprocessing_Data'
        self.___file__ = os.path.abspath(__file__)

        # ---- input size: 2048 ----------
        pass

    def forward(self, x):
        wp = pywt.WaveletPacket(data=x.cpu().data.numpy(), wavelet='db1', mode='symmetric', maxlevel=3)
        x = [wp['aaa'].data, wp['aad'].data, wp['add'].data, wp['ada'].data,
             wp['dda'].data, wp['ddd'].data, wp['dad'].data, wp['daa'].data, ]
        for i in range(len(x)):
            x[i] = torch.tensor(x[i])
        x = torch.cat(x, dim=2).cuda()
        return x


class WPD_MSCNN(nn.Module):
    """ WPD_MSCNN """

    def __init__(self, in_channel=1, out_channel=4):
        super(WPD_MSCNN, self).__init__()
        self.log = 'WPD_MSCNN'
        self.___file__ = os.path.abspath(__file__)
        # input size: 2048
        print('\t\33[07m input size: 2048 \33[0m')
        self.wpd = WPD_Pre_Data()  # 2048-->2048
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=100, stride=2, padding=49, bias=True),  # 2048-->1024
            nn.ReLU(True),  #
            nn.MaxPool1d(2, 2),  # 1024-->512

            nn.Conv1d(16, 32, kernel_size=100, stride=2, padding=49, bias=True),  # 512-->256
            nn.ReLU(True),  #
            nn.MaxPool1d(2, 2),  # 256-->128

            nn.Conv1d(32, 64, kernel_size=66, stride=2, padding=0, bias=True),  # 128-->32
            nn.ReLU(True),  #
            nn.MaxPool1d(2, 2),  # 32-->16
        )
        self.fc = nn.Sequential(
            nn.Linear(64*16, 512),
            nn.ReLU(True),  #
            nn.Dropout(0.5),
            nn.Linear(512, out_channel),
            # nn.Softmax(),
        )

    def forward(self, x):
        x = self.wpd(x)
        x = self.cnn(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x



def debug_net():
    """ 用于调试网络构建代码是否调通 """
    x = torch.rand([3, 1, 2048])
    net = WPD_MSCNN(1, 4)
    print('testing:', net.___file__)
    y = net(x)
    print(y)
    pass










