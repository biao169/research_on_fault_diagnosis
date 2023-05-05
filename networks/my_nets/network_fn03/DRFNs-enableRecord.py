""" 论文中应用的网络，MFPT数据集， """
def enable_DRFNs_01():
    class Max_Feature(nn.Module):
        def __init__(self, in_channel=16, w=128, net='', **kwargs):
            super(Max_Feature, self).__init__()
            # kernel = auto

            self.max_pre = nn.Parameter(torch.rand([in_channel, 1, w // 2], requires_grad=True))  # /(in_channel*w)

            self.kernel_size_predict = Self_Attention(d_in=w, d_k=w * 2, h=2, dropout=0.001)

            self.w_data = None
            pass

        def forward(self, input, kernel_size_train=True, kernel_value_train=True, **kwargs):
            [x, thres] = input
            thres = torch.unsqueeze(thres, 2)
            b, c, w = x.size()
            if not kernel_size_train:
                with torch.no_grad():
                    kernel = self.kernel_size_predict(x)
            else:
                kernel = self.kernel_size_predict(x)
            kernel2 = torch.mean(kernel) + 1e-1  # 归一化kernel，在3~w之间
            kernel = torch.clamp(kernel2 - 1e-1, 0, 1) * (w // 2 - 3) + 3  # torch.tensor(kernel2, dtype=torch.int32)

            w = self.max_pre[:c, :1, :(kernel.type(torch.int32))] / kernel2  # .clone().requires_grad_(True)
            try:
                if not kernel_value_train:
                    with torch.no_grad():
                        y_fitted = functional.conv1d(input=x, weight=w, bias=None, stride=1, padding='same', dilation=1,
                                                     groups=c)
                else:
                    y_fitted = functional.conv1d(input=x, weight=w, bias=None, stride=1, padding='same', dilation=1,
                                                 groups=c)
            except:
                print(kernel2.item(), kernel.item(), w.shape)
                raise
            y_fitted = nn.Sigmoid()(y_fitted)  # 缩放到 [0,1]区间
            x = torch.where(x > thres, x, y_fitted)
            self.w_data = w.clone().detach().requires_grad_(False)
            return x

        def __del__(self):
            if self.w_data is not None:
                print('Max_Feature weight:', self.w_data.shape, '\n', self.w_data[:3, 0, :10], flush=True)
            else:
                print('Max_Feature weight:', self.w_data, flush=True)

    class Thread_Calculate(nn.Module):
        def __init__(self, in_channel=1, out_channel=1, w=128, is_maxpool=False, **kwargs):
            super(Thread_Calculate, self).__init__()
            self.predict = Max_Feature(in_channel, w, **kwargs)
            self.conv = nn.Sequential(
                nn.BatchNorm1d(in_channel),
                nn.Dropout(0.01),  # 0.08
                # nn.ReLU(inplace=True),
                nn.Conv1d(in_channel, in_channel, bias=True, kernel_size=3, stride=1, padding=1),
                # nn.Dropout(0.05),
                # nn.BatchNorm1d(in_channel),
                nn.Conv1d(in_channel, in_channel, bias=True, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm1d(in_channel),
                nn.Sigmoid(),
                # nn.ReLU(inplace=True)

            )
            self.advp = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
            )
            # in_channel = out_channel
            self.fc = nn.Sequential(
                nn.Linear(in_channel, in_channel),
                nn.Linear(in_channel, in_channel),
                nn.Sigmoid(),
            )

            self.out = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm1d(out_channel),
                # nn.MaxPool1d(2, 2),
                nn.Dropout(0.01),
            )
            self.is_maxpool = is_maxpool
            self.maxPool = nn.Sequential(
                nn.Conv1d(out_channel, out_channel, kernel_size=3, stride=2, padding=1),
                nn.AvgPool1d(2, 2),
            )

            pass

        def forward(self, input, **kwargs):
            x = self.conv(input)  # [b,c,w]
            thres = self.advp(x)  # [b,c,1]
            thres = nn.Flatten()(thres)  # [b,c]
            thres = self.fc(thres)  # [b,c]
            pred = self.predict([x, thres], **kwargs)
            pred = nn.Sigmoid()(pred + input)  # x
            pred = self.out(pred)
            if self.is_maxpool:
                pred = self.maxPool(pred)

            return pred
            pass

    class DRFNs_Predict(nn.Module):
        def __init__(self, in_channel=1, out_channel=10, **kwargs):
            super(DRFNs_Predict, self).__init__()
            self.log = 'DRFNs-' + str(kwargs['net']) + ' ' + str('kernel auto')
            self.work_name = 'DRFNs'
            self.___file__ = os.path.abspath(__file__)
            self.cnn = nn.Sequential(
                nn.BatchNorm1d(in_channel),
                nn.Conv1d(in_channel, 6, kernel_size=4, stride=2, padding=1),  # 1024
            )

            self.pred_s = nn.ModuleList([
                Thread_Calculate(6, 8, 1024, True, **kwargs),  # 1024-->
                Thread_Calculate(8, 16, 256, True, **kwargs),  ## 16
                Thread_Calculate(16, 32, 64, False, **kwargs),  ## 8*8
                Thread_Calculate(32, 64, 64, False, **kwargs),  # 4
                Thread_Calculate(64, 128, 64, False, **kwargs),  # 2
                Thread_Calculate(128, 256, 64, False, **kwargs),  # 2
                Thread_Calculate(256, 64, 64, False, **kwargs),  # 2
                Thread_Calculate(64, 64, 64, True, **kwargs),  # 2
                Thread_Calculate(64, 128, 16, True, **kwargs),  # 2
            ])
            out = 128
            self.fc = nn.Sequential(
                nn.Flatten(),
                # nn.BatchNorm1d(out*4),
                nn.Linear(out * 4, 1024, ),
                # nn.ReLU(inplace=True),
                nn.Linear(1024, out_channel),
                nn.Sigmoid()
            )

            pass

        def forward(self, x, **kwargs):
            # print('cnn::', x.shape)
            x = self.cnn(x)  ##  [b,1,2048] --> [b,6,128]
            # x = self.pred_s(x, **kwargs)
            for pred in self.pred_s:
                x = pred(x, **kwargs)

            x = self.fc(x)
            return x

    pass




