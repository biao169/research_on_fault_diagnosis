import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold
import pandas as pd
from PIL import Image
import torch, os
import torch.utils.data
import torchvision
from torch import nn, optim
from torchvision import transforms

global_features = []

''' 网络某层的特征散点分布图 '''


# print(net)
class LayerHook:
    """ 设置hook函数，提取网络指定层的输出值 """
    """ 与直接在forwar中提取等效 """
    features = global_features.copy()

    def __init__(self, model, layer_num):
        self.features.clear()
        self.hook = model[layer_num].register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features.append(output.cpu().detach())
        # print('add:', self.features[-1].size())

    def remove(self):
        self.hook.remove()


def t_sne_add_data(data):
    global_features.append(data)


class t_SNE_Visualization:
    """ 通过一批数据输入，获取对应输出，进行t-sne聚类，然后可视化"""
    X, Y = None, None

    def __init__(self, ):
        self.hook = None
        self.labels_list = []
        global_features.clear()

    def set_hook(self, model, num_layer_idx, ):
        """ 第一步首先设置hook """
        self.hook = LayerHook(model, num_layer_idx)
        pass

    def add_label(self, label):
        """ 训练过程中，记录每一次的label，方便与获取的hook数据对应 """
        self.labels_list.append(label)
        pass

    def __check_data(self):
        """ 将hook和label数据拼接，移动到CPU """
        if self.hook is not None:
            outp = self.hook.features
        else:
            outp = global_features

        if len(self.labels_list) != len(outp):
            print('\033[31m', f'data ERROR: hook\' size is {len(outp)}, label\'s size is {len(self.labels_list)}',
                  '\033[0m', )
            raise ValueError(f'data ERROR: hook\' size is {len(outp)}, label\'s size is {len(self.labels_list)}')

        X = torch.cat(outp, dim=0)
        Y = torch.cat(self.labels_list, dim=0)
        # print((X.shape), len(Y.shape))
        X = X.reshape(X.shape[0], -1)
        Y = Y.reshape(Y.shape[0], -1)
        # print((X.shape), (Y.shape))
        X, Y = X.cpu().detach().numpy(), Y.cpu().detach().numpy()
        self.X, self.Y = X, Y
        pass

    def t_sne(self, X=None):
        """ 计算特征 """
        if X is None: X = self.X
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=200)
        X_tsne = tsne.fit_transform(X)
        # print("Org data dimension is {}. Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

        # '''嵌入空间可视化 归一化 '''
        x_min, x_max = X_tsne.min(0), X_tsne.max(0)
        X_norm = (X_tsne - x_min) / (x_max - x_min)
        X_norm = np.array(X_norm)
        # self.X = X_norm
        return X_norm

    def t_sne_plot(self, dat=None, label=None):
        """ 最后：绘图 """
        self.__check_data()
        if dat is None: dat = self.X
        if label is None: label = self.Y
        # dat = np.random.randn(100,20)
        # label = np.random.randint(0,10,[100, 1])
        # print(label.size)
        maker = ['o', 'v', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']  # 设置散点形状
        colors = ['blue', 'black', 'cyan', 'lime', 'r', 'violet', 'tomato', 'm', 'peru', 'olivedrab',
                  'hotpink', 'yellow']  # 设置散点颜色

        ### 设置字体格式
        font1 = {'family': 'Times New Roman',
                 'weight': 'bold',
                 'size': 10,
                 }

        X = self.t_sne(dat)
        Y = label
        size = max(Y)[0] + 1
        # print('max=', size)
        fig = plt.figure()  ##figsize=(5, 5)
        plt.tight_layout()
        for i in range(size):
            idx = np.argwhere(Y == i)[:, 0]
            # print('==', idx)
            x = X[idx, :]
            plt.scatter(x[:, 0], x[:, 1], s=20, c=colors[i], marker=maker[i],
                        # , cmap='brg' marker=maker[int(Y[i][0])],
                        alpha=0.8, label=str(i)  ## int(Y[i][0])
                        )
        plt.subplots_adjust(top=0.86, left=0.1, right=0.86, bottom=0.1)
        # plt.legend()loc='best',
        plt.legend(scatterpoints=1, markerscale=1,  ## , labelspacing=0.9, columnspacing=1.4
                   bbox_to_anchor=(1.15, 0.98), ncol=1, prop=font1)  # , handletextpad=0.1
        plt.draw()
        plt.pause(0.1)

        # global scatter
        # for i, x in enumerate(X):
        #     # plt.text(x[0], x[1], s=str(Y[i][0]))
        #     scatter = plt.scatter(x[0], x[1], cmap='brg', # s=50, marker=maker[int(Y[i][0])],
        #                 c='', edgecolors=colors[int(Y[i][0])], alpha=0.65, label=int(Y[i][0]))
        #
        # plt.subplots_adjust(left=None, bottom=0.15, right=None, top=None,
        #                     wspace=0.1, hspace=0.15)
        # # plt.legend(scatterpoints=1, labels=maker, loc='best', labelspacing=0.4, columnspacing=0.4, markerscale=2,
        # #            bbox_to_anchor=(0.9, 0), ncol=12, prop=font1, handletextpad=0.1)
        # # plt.legend()
        # plt.legend(handles=scatter.legend_elements()[0],
        #            title="species")
        # # plt.savefig('./'+str(sour)+str(tar)+'.png', format='png',dpi=300, bbox_inches='tight')
        # plt.show(fig)


"""
def example():
    net = torch.nn.Module()  # building network
    dataloader = torch.utils.data.dataloader.DataLoader(None)
    
    tsne = t_SNE_Visualization()
    tsne.set_hook(net.fc, 2)  # set hook in the net
    for i, dat in enumerate(dataloader):  # input data
        ''' Design program '''
        label, inputs = dat
        output = net(inputs)
        tsne.add_label(label)  # get label
    tsne.t_sne_plot()  # draw picture
"""
