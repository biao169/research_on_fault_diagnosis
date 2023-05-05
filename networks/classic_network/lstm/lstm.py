# -*- coding:utf-8 -*-
import torch, os
from torch import nn
import warnings

""" 参考文献
    S. Hao, F.-X. Ge, Y. Li, et al., 
    "Multisensor bearing fault diagnosis based on 
    one-dimensional convolutional long short-term memory networks," 
    Measurement, vol. 159, 2020.
"""
# ----------------------------inputsize >=28-------------------------------------------------------------------------

class CNN_LSM(nn.Module):
    def __init__(self, in_channel=1, out_channel=64):
        super(CNN_LSM, self).__init__()
        self.log = 'CNN_LSM'
        self.___file__ = os.path.abspath(__file__)
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=64, stride=16, padding=24),  # 1024-->64
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2), # 64-->32

            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2), # 32-->32
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2), # 32-->16

            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2), # 16-->16
            nn.ReLU(inplace=False),
            nn.MaxPool1d(kernel_size=2, stride=2), # 16-->8
        )
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        hidden_size, num_layers = 256, 1
        self.lstm1 = nn.LSTM(8*256, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.ch1 = (torch.zeros(num_layers, 128, hidden_size, device=self.device),
                    torch.zeros(num_layers, 128, hidden_size, device=self.device))

        hidden_size, num_layers = 128, 1
        self.lstm2 = nn.LSTM(256, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.ch2 = (torch.zeros(num_layers, 128, hidden_size, device=self.device),
                    torch.zeros(num_layers, 128, hidden_size, device=self.device))

        hidden_size, num_layers = out_channel, 1
        self.lstm3 = nn.LSTM(128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=0.5)
        self.ch3 = (torch.zeros(num_layers, 128, out_channel, device=self.device),
                    torch.zeros(num_layers, 128, out_channel, device=self.device))
        pass

    def get_hc(self, num_layers, batch_size, hidden_size):
        # hidden_size, num_layers = 128, 1
        ch1 = (torch.zeros(num_layers, batch_size, hidden_size, device=self.device),
                    torch.zeros(num_layers, batch_size, hidden_size, device=self.device))
        return ch1

    def forward(self, x):
        x = self.cnn(x)
        # print('== cnn1 ==', x.shape, x.size())
        x = x.view(x.size()[0], 1, -1)
        # print('== cnn2 ==', x.shape, x.size())
        x, ch1 = self.lstm1(x, self.get_hc(1, x.size()[0],  256))
        # print('== cnn3 ==', x.shape, x.size())
        x, ch2 = self.lstm2(x, self.get_hc(1, x.size()[0],  128))
        # print('== cnn4 ==', x.shape, x.size())
        x, ch3 = self.lstm3(x, self.get_hc(1, x.size()[0],  64))
        # print('== cnn5 ==', x.shape, x.size())

        return x

class LSTM_2DC(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(LSTM_2DC, self).__init__()
        self.log = 'LSTM_2DC'
        self.work_name = 'LSTM_2DC'
        self.___file__ = os.path.abspath(__file__)
        ## input w=1024
        self.cnn1 = CNN_LSM(1, 64)
        self.cnn2 = CNN_LSM(1, 64)
        self.fc = nn.Sequential(
            nn.Linear(64*2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channel),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        [x1, x2] = x
        # print('== net1 ==', x1.shape, x2.shape)
        x1 = self.cnn1(x1)
        x2 = self.cnn2(x2)
        # print('== net3 ==', x1.shape, x2.shape)
        x = torch.cat([x1, x2], dim=1)
        # print('== net4 ==', x.shape)
        x = x.view(x.size()[0], -1)
        # print('== net5 ==', x.shape)
        x = self.fc(x)
        return x

class LSTM_1DC(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(LSTM_1DC, self).__init__()
        self.log = 'LSTM_1DC'
        self.work_name = 'LSTM_1DC'
        self.___file__ = os.path.abspath(__file__)
        ## input w=1024
        self.cnn1 = CNN_LSM(1, 64)
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, out_channel),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        [x1, x2] = x
        # print('== net1 ==', x1.shape, x2.shape)
        x = self.cnn1(x1)
        # print('== net3 ==', x1.shape, x2.shape)
        # print('== net4 ==', x.shape)
        x = x.view(x.size()[0], -1)
        # print('== net5 ==', x.shape)
        x = self.fc(x)
        return x