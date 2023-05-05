# -*- coding:utf-8 -*-
import torch, os
from torch import nn
import warnings


# ----------------------------inputsize >=28-------------------------------------------------------------------------
class WDCNN(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(WDCNN, self).__init__()
        self.log = 'WDCNN'
        self.work_name = 'WDCNN'
        self.___file__ = os.path.abspath(__file__)


        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64,stride=16,padding=24),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),

            nn.MaxPool1d(kernel_size=2,stride=2)
            )

        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=3,padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3,padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )  # 32, 12,12     (24-2) /2 +1

        self.layer5 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
            # nn.AdaptiveMaxPool1d(4)
        )  # 32, 12,12     (24-2) /2 +1

        self.fc=nn.Sequential(
            nn.Linear(192, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Linear(100, out_channel),
            # nn.Softmax(dim=1)
        )


    def forward(self, x):
        # print(x.shape)
        x = self.layer1(x) #[16 64]
        # print(x.shape)
        x = self.layer2(x)  #[32 124]
        # print(x.shape)
        x = self.layer3(x)#[64 61]
        # print(x.shape)
        x = self.layer4(x)#[64 29]
        # print(x.shape)
        x = self.layer5(x)#[64 13]
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print('===net===', x.shape)
        x = self.fc(x)
        return x

