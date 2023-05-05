# -*- coding:utf-8 -*-
import torch, os
from torch import nn
import warnings

""" 参考文献
    C. -Y. Lee and G. -L. Zhuo, "Identifying Bearing Faults Using Multiscale Residual Attention and Multichannel Neural Network," in IEEE Access, vol. 11, pp. 26953-26963, 2023, doi: 10.1109/ACCESS.2023.3257101.
基于振动信号，用于强噪声下的故障诊断，变转速下的故障诊断
"""


class ChannelAttention(nn.Module):
    """ 通道注意力机制 """

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_MLP = nn.Sequential(
            nn.Conv1d(in_planes, in_planes, kernel_size=1, bias=True),  # // ratio
            # nn.ReLU(),
            # nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.shared_MLP(self.avg_pool(x))  # self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # max_out = self.shared_MLP(self.max_pool(x))  # self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out  # + max_out
        return self.sigmoid(out)*x


class SpatialAttention(nn.Module):
    """ 空间注意力机制 """

    def __init__(self, kernel_size=7, in_channel=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Sequential(
            nn.BatchNorm1d(in_channel),
            nn.Conv1d(in_channel, in_channel, kernel_size, padding=padding, bias=True)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # x = torch.cat([avg_out, max_out], dim=1)
        # x = self.conv1(x)
        out = self.conv1(x)
        return self.sigmoid(out) * x


class MFE(nn.Module):
    """ MULTI-SCALE FEATURE EXTRACTION MODULE """

    def __init__(self, in_channel=1, out_channel=64, n_scale=4, kernel=4, stride=2, in_w=1024, out_w=None, ratio=0.5):
        super(MFE, self).__init__()
        self.log = 'MSCNet_MFE'
        self.___file__ = os.path.abspath(__file__)
        if not out_w: out_w = in_w*ratio
        padding = ((out_w-1)*stride-in_w+kernel)//2
        out_w = (in_w-kernel+2*padding)/stride+1
        print(f'MFE conv padding is {padding}, and output size is {out_w}!')
        self.w_cnn = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel_size=kernel, stride=stride, padding=int(padding)),  # 1024-->64
        )
        if out_channel % n_scale != 0: raise ValueError(
            f'out_channel={out_channel},  n_scale={n_scale}, out_channel%n_scale!=0')
        self.n_scale = n_scale
        self.n_scale_nets = []
        for i in range(n_scale):
            k = 2 * (i + 1) + 1
            self.n_scale_nets += [
                nn.Conv1d(out_channel // n_scale, out_channel // n_scale, kernel_size=k,  # stride=stride,
                          padding='same')]
        self.n_scale_nets = nn.ModuleList(self.n_scale_nets)

        self.out = nn.Sequential(
            nn.BatchNorm1d(out_channel),  #
            nn.ReLU(inplace=False),
        )

        pass

    def train_scale(self, x):
        b, c = x.shape[:2]
        ym_out = 0
        ym = 0
        step = c // self.n_scale
        for i, scale in enumerate(self.n_scale_nets):
            yw = x[:b, i * step: (i + 1) * step]
            if i == 0:
                ym = scale(yw)
                ym_out = ym.clone()
            else:
                yw = yw + ym
                ym = scale(yw)
                ym_out = torch.cat([ym_out, ym], dim=1)
        return ym_out

    def forward(self, x):
        x = self.w_cnn(x)
        x = self.train_scale(x)
        x = self.out(x)
        return x


class RAFE(nn.Module):
    """ RESIDUAL ATTENTION FEATURE EXTRACTION MODULE """

    def __init__(self, in_channel=1, out_channel=64, group=4):
        super(RAFE, self).__init__()
        self.log = 'MSCNet_RAFE'
        self.___file__ = os.path.abspath(__file__)
        self.group = group
        self.channel_attention = ChannelAttention(in_channel//group//2)
        self.spatial_attention = SpatialAttention(in_channel=in_channel//group//2, kernel_size=3)
        pass

    def train_attention(self, x):
        b, c = x.shape[:2]
        x_out = None
        step = c // self.group
        chan_s = step // 2
        for i in range(self.group):
            brand_channel = x[:b, i * step:(i + 1) * step - chan_s]
            x_chan = self.channel_attention(brand_channel)
            brand_spatial = x[:b, (i + 1) * step - chan_s:(i + 1) * step]
            x_spat = self.spatial_attention(brand_spatial)
            if x_out is None: x_out = torch.cat([x_chan, x_spat], dim=1)
            else: x_out = torch.cat([x_out, x_chan, x_spat], dim=1)
        return x_out

    def forward(self, x):
        ya = self.train_attention(x)
        x = ya * x + x
        return x


""" 论文使用的数据预处理方法 """
# waight = torch.ones([3])
# def morphological_filter(x):
#     """ 形态学滤波 """
#
#     dilation = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
#     dilation.weight = torch.ones([3])
#
#
#     return torch.nn.functional.conv1d(x, waight, stride=1, padding=1)
#
#
# def mean_filter(x):
#     """ 均值滤波 """
#     return torch.nn.functional.conv1d(x, waight, stride=1, padding=1)

class MSCNet(nn.Module):
    def __init__(self, in_channel=1, out_channel=10):
        super(MSCNet, self).__init__()
        self.log = 'MSCNet'
        self.work_name = 'MSCNet'
        self.___file__ = os.path.abspath(__file__)
        ## input size: 1024
        print('\33[07m \tinput size: 1024\33[0m')
        self.link_1 = nn.Sequential(
            MFE(1, 16, n_scale=2, kernel=96, stride=2, in_w=1024, ratio=0.5),
            RAFE(16, 0, group=2),
            MFE(16, 32, n_scale=2, kernel=48, stride=2, in_w=512, ratio=0.5),
            RAFE(32, 0, group=2),
            MFE(32, 64, n_scale=2, kernel=24, stride=2, in_w=256, ratio=0.5),
            RAFE(64, 0, group=2),
        )
        self.link_2 = nn.Sequential(
            MFE(1, 16, n_scale=2, kernel=96, stride=2, in_w=1024, ratio=0.5),
            RAFE(16, 0, group=2),
            MFE(16, 32, n_scale=2, kernel=48, stride=2, in_w=512, ratio=0.5),
            RAFE(32, 0, group=2),
            MFE(32, 64, n_scale=2, kernel=24, stride=2, in_w=256, ratio=0.5),
            RAFE(64, 0, group=2),
        )
        self.GAP = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64*2, out_channel),
        )

        """ 论文提及的数据预处理：形态学运算和均值滤波 """
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.maxpool.weight = torch.ones([3])
        self.waight = torch.ones([in_channel, 1, 3]) / 3

        if torch.cuda.is_available(): self.use_gpu()


    def preprocessing_morphological_filter(self, x):
        def dilation(dat):
            return self.maxpool(dat)

        def erosion(dat):
            """ 需要注意：输入dat是不是全为正数，是否应该先取绝对值 """
            return -self.maxpool(-dat)

        closing = erosion(dilation(x))
        opening = dilation(erosion(x))
        scth = closing - opening
        return scth
        pass

    def preprocessing_mean_filter(self, x):
        return torch.nn.functional.conv1d(x, self.waight, stride=1, padding=1)

    def use_gpu(self):
        self.maxpool.cuda()
        self.waight.cuda()

    def forward(self, x):
        x1 = self.link_1(self.preprocessing_mean_filter(x))
        x2 = self.link_2(self.preprocessing_morphological_filter(x))
        x1 = self.GAP(x1)
        x2 = self.GAP(x2)
        x1 = torch.cat([x1, x2], dim=1).squeeze(dim=-1)
        x1 = self.fc(x1)
        return x1


def debug_net():
    """ 用于调试网络构建代码是否调通 """
    x = torch.rand([3, 1, 1024])
    net = MSCNet(1, 10)
    print('testing:', net.___file__)
    y = net(x)
    pass