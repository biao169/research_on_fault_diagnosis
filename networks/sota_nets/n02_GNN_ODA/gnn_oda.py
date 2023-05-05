# -*- coding:utf-8 -*-
__AUTHOR__ = 'Jinbiao Tan'
"""
    C. -B. Chou and C. -H. Lee, "Generative Neural Network-Based Online Domain Adaptation (GNN-ODA) Approach for Incomplete Target Domain Data," in IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1-10, 2023, Art no. 3508110, doi: 10.1109/TIM.2023.3246495.
    域迁移：训练数据的工况与测试数据的工况不同，通过生成网络生成假图片，补充测试中的工况数据
"""


import torch, os
from torch import nn
import warnings



class Extractor(nn.Module):
    """ Extractor """

    def __init__(self, in_channel=1, out_channel=100):
        super(Extractor, self).__init__()
        self.log = 'GNN_ODA_Extractor'
        self.___file__ = os.path.abspath(__file__)

        # ---- input size: 256 ----------
        print('\33[07m \tinput size: 256\33[0m')
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channel, 8, kernel_size=32, stride=2, padding=0),  # 256-->113
            nn.ReLU(True),
            nn.MaxPool1d(2,2),
            nn.Conv1d(8, 16, kernel_size=4, stride=2, padding=0),  # 56-->27
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, kernel_size=4, stride=1, padding=0),  # 27-->10
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),  # --->5
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.out = nn.Sequential(

            nn.Linear(160, 128),
            nn.ReLU(True),  #
            nn.Linear(128, out_channel),
            nn.ReLU(inplace=True),
        )

        pass

    def forward(self, x):
        x = self.cnn(x)
        x = self.flatten(x)
        x = self.out(x)
        return x


class Generator(nn.Module):
    """ Generator """

    def __init__(self, in_channel=1, out_channel=100):
        super(Generator, self).__init__()
        self.log = 'GNN_ODA_Generator'
        self.___file__ = os.path.abspath(__file__)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channel, 8, kernel_size=5, stride=1, padding=0),  # 100-->96
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(8, 16, kernel_size=4, stride=1, padding=0),  # 96-->45
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),
            nn.Conv1d(16, 32, kernel_size=4, stride=1, padding=0),   # 45-->19
            nn.ReLU(True),
            nn.MaxPool1d(2, 2),  # 19 --> 9
        )
        self.flatten = nn.Flatten(start_dim=1)
        self.out = nn.Sequential(

            nn.Linear(288, 128),  # 32*9 --> 128
            nn.ReLU(True),  #
            nn.Linear(128, out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.cnn(x)
        # print('===cnn 1===', x.shape)
        x = self.flatten(x)
        # print('===cnn 2===', x.shape)
        x = self.out(x)
        return x


class Classifier(nn.Module):
    """ Classifier """

    def __init__(self, in_channel=100, out_channel=2):
        super(Classifier, self).__init__()
        self.log = 'GNN_ODA_Classifier'
        self.___file__ = os.path.abspath(__file__)

        self.flatten = nn.Flatten(start_dim=1)
        self.out = nn.Sequential(
            nn.Linear(in_channel, 64),
            nn.ReLU(True),  #
            nn.Linear(64, 20),
            nn.ReLU(inplace=True),
            nn.Linear(20, out_channel),
            # nn.Softmax(),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.out(x)
        return x


class GNN_ODA(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(GNN_ODA, self).__init__()
        self.log = 'GNN_ODA main class'
        self.work_name = 'GNN_ODA'
        self.___file__ = os.path.abspath(__file__)
        # -------- input w= 256  --------------
        self.extractor = Extractor(1, 100)
        self.generator = Generator(1, 100)
        self.classifier = Classifier(100, 2)

    def forward(self, x):
        x = self.extractor(x)
        x1 = x.unsqueeze(dim=1)
        # print('===gnn 2===', x.shape, x1.shape)
        x2 = self.generator(x1)
        out1 = self.classifier(x)
        out2 = self.classifier(x2)
        return x, x2, out1, out2


def debug_net():
    """ 用于调试网络构建代码是否调通 """
    x = torch.rand([3, 1, 256])
    net = GNN_ODA(1, 10)
    print('testing:', net.___file__)
    y = net(x)
    print(y)
    pass










