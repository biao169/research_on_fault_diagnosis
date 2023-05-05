# -*- coding:utf-8 -*-
import torch, os
from torch import nn
import warnings

""" 参考文献
    S. Shao, R. Yan, Y. Lu, P. Wang and R. X. Gao, 
    "DCNN-Based Multi-Signal Induction Motor Fault Diagnosis,"
     in IEEE Transactions on Instrumentation and Measurement, 
     vol. 69, no. 6, pp. 2658-2669, June 2020, doi: 10.1109/TIM.2019.2925247.
"""
# ----------------------------inputsize >=28-------------------------------------------------------------------------

class DCNN_MutilChannel(nn.Module):
    def __init__(self, in_channel=2, out_channel=10):
        super(DCNN_MutilChannel, self).__init__()
        self.log = 'DCNN_MutilChannel'
        self.___file__ = os.path.abspath(__file__)
        ## input w=128*128
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channel, 12, kernel_size=6, stride=1, padding=2),  # 128-->127
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 127-->63

            nn.Conv2d(12, 24, kernel_size=8, stride=1, padding=3), # 63-->62
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 63-->31

            nn.Conv2d(24, 24, kernel_size=6, stride=1, padding=2), # 31-->30
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 30-->15
        )
        self.fc = nn.Sequential(
            nn.Linear(15*15*24, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_channel),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.cnn(x)  # [2, 15,15]
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print('===net===', x.shape)
        x = self.fc(x)
        return x

class DCNN_Merged(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(DCNN_Merged, self).__init__()
        self.log = 'DCNN_Merged'
        self.___file__ = os.path.abspath(__file__)
        ## input w=128*128

        self.cnn1 = nn.Sequential(
            nn.Conv2d(in_channel, 6, kernel_size=6, stride=1, padding=2), #  128-->127
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #  127-->63

            nn.Conv2d(6, 12, kernel_size=8, stride=3, padding=1), #  63-->62
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #  62-->31

            nn.Conv2d(12, 12, kernel_size=6, stride=3, padding=2), #  31-->30
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #  30-->15

        )
        # self.fc1 = nn.Sequential(
        #     nn.Linear(1*15*15*12, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, out_channel),
        #     # nn.Softmax(dim=1)
        # )

        self.cnn2 = nn.Sequential(
            nn.Conv2d(in_channel, 6, kernel_size=3, stride=1, padding=1), #  128-->128
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #  128-->64

            nn.Conv2d(6, 12, kernel_size=3, stride=1, padding=1), #  64-->64
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #  64-->32

            nn.Conv2d(12, 12, kernel_size=6, stride=1, padding=1), #  32-->32
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), #  32-->16
        )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(1*16*16*12, 512),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(512, out_channel),
        #     nn.Softmax(dim=1)
        # )

        self.fc = nn.Sequential(
            nn.Linear(15*15*12+16*16*12, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, out_channel),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        [x1, x2] = x
        x1 = self.cnn1(x1)
        # print(x.shape)
        x2 = self.cnn1(x2)
        x1 = x.view(x1.size(0), -1)
        x2 = x.view(x2.size(0), -1)
        x = torch.cat([x1,x2], dim=1)
        # print('===net===', x.shape)
        x = self.fc(x)
        return x