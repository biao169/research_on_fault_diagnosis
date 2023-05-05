import os

import torch
import torch.nn as nn
from torch.nn import Linear, Conv1d, Conv2d, ReLU, BatchNorm1d, AvgPool1d, AdaptiveAvgPool1d, Flatten, Sigmoid, Softmax


def calculator_outSize(insize, kernel, padding, stride):
    return (insize - kernel + 2 * padding) // stride + 1


def calculator_padding(insize, outsize, kernel, stride):
    return ((outsize - 1) * stride - insize + kernel) / 2


class RSBU_CW(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
        super(RSBU_CW, self).__init__()
        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            Linear(in_features=out_channels, out_features=out_channels),
            BatchNorm1d(out_channels),
            ReLU(inplace=True),
            Linear(in_features=out_channels, out_features=out_channels),
            Sigmoid()
        )
        self.flatten = Flatten()
        self.average_pool = AvgPool1d(kernel_size=1, stride=2)



    def forward(self, input):
        x = self.BRC(input)
        x_abs = torch.abs(x)
        gap = self.global_average_pool(x_abs)
        gap = self.flatten(gap)
        alpha = self.FC(gap)
        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)

        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x), n_sub)

        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)

        """ 通道只能增加， 不能减少 """
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            # B,C,W = input.shape ## 1D没有H
            size = list(input.shape)
            if (self.out_channels - self.in_channels)%2 ==0:
                size[1] = (self.out_channels - self.in_channels)//2
                zero_padding = torch.zeros(size, device=input.device)
                # 官方是上下叠加，而不是加在最好
                input = torch.cat([zero_padding, input, zero_padding], dim=1)
            else:
                size[1] = (self.out_channels - self.in_channels) // 2
                zero_padding = torch.zeros(size)
                size[1] = (self.out_channels - self.in_channels) // 2 +1
                zero_padding2 = torch.zeros(size, device=input.device)
                input = torch.cat([zero_padding, input, zero_padding2], dim=1)

        result = x + input

        return result


class RSBU_CS(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, down_sample=False):
        super(RSBU_CS, self).__init__()

        self.down_sample = down_sample
        self.in_channels = in_channels
        self.out_channels = out_channels
        stride = 1
        if down_sample:
            stride = 2
        self.BRC = nn.Sequential(
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=1)
        )
        self.global_average_pool = nn.AdaptiveAvgPool1d(1)
        self.FC = nn.Sequential(
            Linear(in_features=out_channels, out_features=out_channels),
            BatchNorm1d(out_channels),
            ReLU(inplace=True),
            Linear(in_features=out_channels, out_features=1),
            Sigmoid()
        )
        self.flatten = Flatten()
        self.average_pool = AvgPool1d(kernel_size=1, stride=2)



    def forward(self, input:torch.Tensor):
        x = self.BRC(input)
        x_abs = torch.abs(x)
        gap = self.global_average_pool(x_abs)
        gap = self.flatten(gap)
        alpha = self.FC(gap)
        gap = self.global_average_pool(gap)  ##  CS 和 CW的区别地方
        threshold = torch.mul(gap, alpha)
        threshold = torch.unsqueeze(threshold, 2)

        # 软阈值化
        sub = x_abs - threshold
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x), n_sub)

        if self.down_sample:  # 如果是下采样，则对输入进行平均池化下采样
            input = self.average_pool(input)

        """ 通道只能增加， 不能减少 """
        if self.in_channels != self.out_channels:  # 如果输入的通道和输出的通道不一致，则进行padding,直接通过复制拼接矩阵进行padding,原代码是通过填充0
            # B,C,W = input.shape ## 1D没有H
            size = list(input.shape)
            if (self.out_channels - self.in_channels)%2 ==0:
                size[1] = (self.out_channels - self.in_channels)//2
                zero_padding = torch.zeros(size, device=input.device)
                # 官方是上下叠加，而不是加在最好
                input = torch.cat([zero_padding, input, zero_padding], dim=1)
            else:
                size[1] = (self.out_channels - self.in_channels) // 2
                zero_padding = torch.zeros(size, device=input.device)
                size[1] = (self.out_channels - self.in_channels) // 2 +1
                zero_padding2 = torch.zeros(size, device=input.device)
                input = torch.cat([zero_padding, input, zero_padding2], dim=1)

        result = x + input

        return result

class DRSNet_CW(torch.nn.Module):
    def __init__(self, in_channel=1, out_channel=8):
        super(DRSNet_CW, self).__init__()
        self.log = 'DRSNet_CW'
        self.work_name= 'DRSNet_CW'
        self.___file__ = os.path.abspath(__file__)
        self.conv1 = nn.Sequential(
                        Conv1d(in_channels=in_channel, out_channels=4, kernel_size=3, stride=2, padding=1),
                    )
        self.rsbu_cw = nn.Sequential(
                        RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=True),
                        RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=False),
                        RSBU_CW(in_channels=4, out_channels=8, kernel_size=3, down_sample=True),
                        RSBU_CW(in_channels=8, out_channels=8, kernel_size=3, down_sample=False),
                        RSBU_CW(in_channels=8, out_channels=16, kernel_size=3, down_sample=True),
                        RSBU_CW(in_channels=16, out_channels=16, kernel_size=3, down_sample=False),
                    )
        self.classify_fc = nn.Sequential(
                        BatchNorm1d(16),
                        ReLU(inplace=True),
                        AdaptiveAvgPool1d(1),
                        Flatten(),
                        Linear(16, out_channel),
                    )


    def forward(self, x):  # 1*256
        x = self.conv1(x)
        x = self.rsbu_cw(x)
        x = self.classify_fc(x)
        return x

class DRSNet_CS(torch.nn.Module):
    def __init__(self, in_channel=1, out_channel=8):
        super(DRSNet_CS, self).__init__()
        self.log = 'DRSNet_CS'
        self.___file__ = os.path.abspath(__file__)

        self.conv1 = nn.Sequential(
                        Conv1d(in_channels=in_channel, out_channels=4, kernel_size=3, stride=2, padding=1),
                    )
        self.rsbu_cw = nn.Sequential(
                        RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=True),
                        RSBU_CW(in_channels=4, out_channels=4, kernel_size=3, down_sample=False),
                        RSBU_CW(in_channels=4, out_channels=8, kernel_size=3, down_sample=True),
                        RSBU_CW(in_channels=8, out_channels=8, kernel_size=3, down_sample=False),
                        RSBU_CW(in_channels=8, out_channels=16, kernel_size=3, down_sample=True),
                        RSBU_CW(in_channels=16, out_channels=16, kernel_size=3, down_sample=False),
                    )
        self.classify_fc = nn.Sequential(
                        BatchNorm1d(16),
                        ReLU(inplace=True),
                        AdaptiveAvgPool1d(1),
                        Flatten(),
                        Linear(16, out_channel),
                    )


    def forward(self, x):
        x = self.conv1(x)
        x = self.rsbu_cw(x)
        x = self.classify_fc(x)
        return x

if __name__ == '__main__':
    # calculator_padding()
    x= torch.arange(0,32*16*2, dtype=torch.float).reshape(-1,16,32)

    ds = RSBU_CW(16, 108, 3, True)
    ds(x)
    print('\n=====================================================\n')
    ds = RSBU_CS(16, 32, 3, True)
    ds(x)


    pass







































