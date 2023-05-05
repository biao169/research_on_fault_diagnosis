import os
import pickle

import scipy.io as scio
import pandas as pd
import numpy as np
import torch
import torch.nn.functional
from torch.utils import data
from utils_tool import signal_normalization

""" 数据集来自都灵理工大学的轴承数据DIRG """


def seq_chunk(seq, chunk, step=None, log=None, ishow=False):
    """
    序列阶段：定义一个函数，该函数主要用于将序列数据按照步长和窗口进行序列提取
    """
    if step is None or step == 0:
        step = chunk
    seq_length = seq.shape[0]
    rows = (seq_length - chunk) / step + 1
    if rows - int(rows) != 0 and log is not None:
        if ishow: print('\t', log, '\'s Data cannot be aligned!')
    if chunk == step:  ## 没有重复内容，用reshape更快
        if rows - int(rows) != 0: seq = seq[:int(rows) * chunk]  # 数据过长就丢弃
        new_set = np.array(seq).reshape([-1, chunk])
        return new_set
    rows = int(rows)
    new_set = np.empty(shape=(rows, chunk))
    for i in range(0, seq_length - chunk + 1, step):
        new_set[i // step] = seq[i:i + chunk]
    return new_set


def dict_add_items(a: dict, b: dict):
    aa = a.copy()
    for n in b.keys():
        aa[n] = b[n]
    return aa


class DataSet_DIRG(data.Dataset):
    """ -----  数据集文件夹简介  ---------
        文件使用官方数据集（文件名末尾带编号）
        数据的文件名格式为: CnA_fff_vvv_m.mat
            C: 文件名的根，所有文件通用;
            n: 从0到6的整数值，表示缺陷种类，如0A, 1A，…，6A(表3);
            fff: 100 ~ 500的整数值，表示轴的公称转速(Hz);
            vvv: 称重传感器电压对应的整数值(mV)，表示施加的负载;
            M: 整数值，表示测量是否重复(M =2)或是否重复(M =1);

    """
    """ 统一读取所有文件后使用 data.random_split 划分训练集和测试集，会导致各类的数量不均，
        应该独立对每个类使用data.random_split划分数据集 
    """

    def __init__(self, root, load_files=None, repeat_win=1.0, window=2048, label: int = None, num_class: int = None,
                 train_test_rate: float = 0.7, cate: str = None):
        super(DataSet_DIRG, self).__init__()
        # root = root  # os.path.join(root, folder)
        load_files0 = [
            'C0A_100_505_1.mat',
            'C1A_100_502_2.mat',
            'C2A_100_506_1.mat',
            'C3A_100_505_1.mat',
            'C4A_100_496_1.mat',
            'C5A_100_498_1.mat',
            'C6A_100_500_1.mat',
        ]
        if load_files is None: load_files = load_files0
        files = [os.path.join(root, f) for f in load_files]

        # datalist, numdict = self.load_file_onlyone(files[0], window, repeat_win, False,
        #                                            ishow=True)  ## 为了拼接，cut_same=true,但，减少重复工作

        self.all_data_arr_x_y_z_label = self.load_files_and_format(files, win=window, repeat_win=repeat_win,
                                                                      cut_same=True, label=label, num_class=num_class,
                                                                      ishow=False)
        """ 数据集内容变更"""
        if cate is not None:
            length = len(self.all_data_arr_x_y_z_label[1])
            depart = int(length * train_test_rate)
            if cate.lower().startswith('train'):
                self.all_data_arr_x_y_z_label = [self.all_data_arr_x_y_z_label[0][:depart],
                                                    self.all_data_arr_x_y_z_label[1][:depart]]
            else:
                self.all_data_arr_x_y_z_label = [self.all_data_arr_x_y_z_label[0][depart:],
                                                    self.all_data_arr_x_y_z_label[1][depart:]]

        self.args = {
            'program name': 'DataSet_DIRG',
            'program path': os.path.abspath(__file__),
            'data path': str(root),
            'load files': str(files),
            'repeat_win': repeat_win,
            'window': window,
            'num_class': num_class,
            'log': 'All file\'s data were uniformly normalized!',
        }
        # self.idx = np.arange(0, self.all_data_arr_x_y_z_label[1].shape[0])

        ## 制作缓存文件
        # cached_file = os.path.join(self.root, 'processed_cached.pkl')
        # self.data_file = cached_file
        # if os.path.exists(cached_file) and not rebuild:
        #     print('Cached file exist! %s' % cached_file)
        #     pass
        # else:
        #     data = self.preprocessing_data_in_folder()
        #     with open(cached_file, "wb") as f:
        #         pickle.dump(data, f)
        #     print("Save Cached file: {}".format(cached_file))
        #     self.preprocessing_data_normalization(time_norm=True, vibr_norm=True, temp_norm=False, filename=cached_file)
        # self.data_list = self.load_cached(False)
        pass

    ''' 加载一个文件，提取数据 '''

    def load_file_onlyone(self, path: str, win: int, repeat_win: float, cut_same=False, ishow=False):
        """ 官方数据中有存在放置错误的情况 """
        filename = os.path.basename(path)
        num_name = filename[:-4]  #  .split('_')[-1]
        # print(num_name)
        """ 特殊情况 """
        # if filename == '12k_Drive_End_B028_0_3005.mat': num_name = '048'

        data = scio.loadmat(path)
        num_record = {'origal': {}, 'new': {}}
        tmplist = []  # 0:x  1:y   2:z  3:x  4:y   5:z

        data = data[num_name]
        for i in range(6):
            tmp = data[:,i]
            tmp_chunk = seq_chunk(tmp, chunk=win, step=int(win * (1 - repeat_win)),
                                  log=path + '  ' + str(i),
                                  ishow=ishow)
            tmplist.append(tmp_chunk)
            num_record['origal']['%d'%i] = tmp.shape[0]
            num_record['new']['%d'%i] = tmp_chunk.shape[0]

        if cut_same:
            minnum = np.array(list(num_record['new'].values())).min(axis=0)
            for i in range(len(tmplist)):
                if tmplist[i].shape[0] > minnum:
                    tmplist[i] = tmplist[i][:minnum, :]
        # if len(tmplist) < 3: tmplist.append([])
        return tmplist, num_record

    ''' 加载所有文件（files），缺失列用0补齐 '''

    def load_files_and_format(self, files: [], win: int, repeat_win: float, cut_same=False,
                              label: int = None, num_class: int = None, ishow=False):
        all_data_arr_DE_FE_BA = np.array([])
        all_data_label = np.array([])
        if num_class is None: num_class = len(files)
        for idx, file in enumerate(files):
            datalist, numdict = self.load_file_onlyone(file, win, repeat_win, False,
                                                       ishow=ishow)  ## 为了拼接，cut_same=true,但，减少重复工作

            minnum = np.array(list(numdict['new'].values())).min(axis=0)
            maxnum = np.array(list(numdict['new'].values())).max(axis=0)
            # print('numdict:', file, len(datalist), minnum, numdict )
            if minnum!=maxnum:
                for i in range(len(datalist)):
                    datalist[i]=datalist[i][:minnum]
            if label is None:
                label2 = idx
            else:
                label2 = label

            label_arr = (np.ones(shape=[minnum, 1], dtype=np.int) * label2)  ## 创建该文件对应的 label
            # label_arr = (np.zeros(shape=[minnum, num_class], dtype=np.int)) ## 分类问题需要独热编码
            # label_arr[:,label2] = 1 ## 分类问题需要独热编码

            if not cut_same: raise ValueError("all data should be the same length! [cut_same=true]")
            datalist = np.stack(datalist, axis=2)

            if all_data_arr_DE_FE_BA.any():
                all_data_arr_DE_FE_BA = np.vstack([all_data_arr_DE_FE_BA, datalist])
                all_data_label = np.vstack([all_data_label, label_arr])
            else:
                all_data_arr_DE_FE_BA = datalist
                all_data_label = label_arr
            # print('data shape log:', all_data_label[-1], all_data_arr_DE_FE_BA.shape, all_data_label.shape, datalist.shape[0])

        # all_data_label = torch.nn.functional.one_hot(all_data_label) ## 分类问题需要独热编码
        return all_data_arr_DE_FE_BA, all_data_label

    def __getitem__(self, item):
        """ 使用 data.random_split 后，这里报错不会有提醒 """
        dat, label = self.all_data_arr_x_y_z_label[0], self.all_data_arr_x_y_z_label[1]
        x1, y1, z1, x2, y2, z2 = dat[item, :, 0], dat[item, :, 1], dat[item, :, 2], dat[item, :, 3], dat[item, :, 4], dat[item, :, 5],
        label = label[item][0]
        """ signal processing """
        # DE = signal_normalization.add_gauss_noise(DE, 1)

        dat1 = torch.tensor(y1, dtype=torch.float).reshape(1, -1)
        dat2 = torch.tensor(y2, dtype=torch.float).reshape(1, -1)
        label = torch.tensor(label, dtype=torch.long)  # .reshape(1)
        return label, dat1, dat2  # self.idx[item]
        pass

    def __len__(self):
        return self.all_data_arr_x_y_z_label[1].shape[0]

    def config_make(self):
        return self.args
        pass

    # def collate_fn(self, data):
    #     return data


class DataSet_DIRG_Signal_Process(DataSet_DIRG):
    """ -----  数据集文件夹简介  ---------
        文件使用官方数据集（文件名末尾带编号）
        每个 .mat 文件就是一种故障类型，里边包含两种或三种信号数据
        示例：12k_Drive_End_B007_1_119.mat
            12K: 采样频率；
            Drive_End: 驱动端加速度数据   // Fan_End: 风扇端加速度数据
            B007: 故障位置B和类型007  //  B/IR/OR
            1: 工况条件  // 1797 / 1772 / 1750 / 1730
            119：对应官方下载文件的文件名（编号）
    """
    """ 
        数据提取时，加入信号处理操作
    """

    def __init__(self, root, load_files=None, repeat_win=1.0, window=2048, label: int = None, num_class: int = None,
                 train_test_rate: float = 0.7, cate: str = None, mask_win: int = None, mask_start_idx: int = None,
                 snr: float = None):
        super(DataSet_DIRG_Signal_Process, self).__init__(root, load_files, repeat_win, window, label, num_class,
                                                          train_test_rate, cate)

        self.mask_win = mask_win
        self.mask_start_idx = mask_start_idx
        self.snr = snr
        self.args['program name'] = 'DataSet_CRWU_Signal_Process'
        mask_win = round(mask_win / window, 1) if mask_win is not None else 'random'
        mask_start_idx = round(mask_start_idx / window, 1) if mask_start_idx is not None else 'random'
        snr = round(snr / window, 1) if snr is not None else 'random'
        self.args['signal enhance'] = dict(mask_win=mask_win, mask_start_idx=mask_start_idx, SNR=snr)
        # print('DataSet_CRWU_Signal_Process:', self.mask_win , self.mask_start_idx  )

    def __getitem__(self, item):
        """ 使用 data.random_split 后，这里报错不会有提醒 """
        dat, label = self.all_data_arr_X_Y_Z_label[0], self.all_data_arr_X_Y_Z_label[1]
        x1, y1, z1, x2, y2, z2 = dat[item, :, 0], dat[item, :, 1], dat[item, :, 2], dat[item, :, 3], dat[item, :, 4], dat[item, :, 5],
        label = label[item][0]
        # print(DE.shape, FE.shape, BA.shape, label.shape)  ## (1000,) (1000,) (1000,) (241,)
        """ signal processing """  # Fixme
        y1, real_rate, start_rate = self.signal_processing(y1, np.random.randint(0, 3))
        # if real_rate<0.5:
        #     label = 10
        # print('====real_rate:', real_rate, label)
        dat1 = torch.tensor(y1, dtype=torch.float).reshape(1, -1)
        dat2 = torch.tensor(y2, dtype=torch.float).reshape(1, -1)

        label = torch.tensor(label, dtype=torch.long)  # .reshape(1)
        rate = torch.tensor(real_rate, dtype=torch.float).reshape(1)  #
        return label, rate, start_rate, dat1, dat2,    # self.idx[item]
        pass

    def signal_processing(self, dat, idx):
        if self.mask_win is not None and self.mask_start_idx is not None:
            dat, rate, start_rate = signal_normalization.signal_mask_random(dat, start_idx=self.mask_start_idx,
                                                                            mask_win=self.mask_win)
            if self.snr is not None:
                dat = signal_normalization.add_gauss_noise(dat, self.snr)
            return dat, rate, start_rate
        rate = 1.0
        start_rate = 1.0
        level = np.random.randint(-4, 10)
        if self.snr is not None: level = self.snr
        if idx == 0:
            dat = signal_normalization.add_gauss_noise(dat, level)
            rate = 1.0
            start_rate = 1.0
        elif idx == 1:
            dat, rate, start_rate = signal_normalization.signal_mask_random(dat, start_idx=self.mask_start_idx,
                                                                            mask_win=self.mask_win)
            dat = signal_normalization.add_gauss_noise(dat, level)
            # print('===2==', rate, start_rate)
        return dat, rate, start_rate


class Sub_Little_Dataset(data.Dataset):  ##
    """ 使用 data.random_split 划分数据集后的 dataset 不具备类的功能
        在train_ways/base.py中无法调用 args 的属性，需要重构
        *** data.random_split 划分数据集，各类样本可能不均衡 ***
    """

    def __init__(self, dataset, args):
        self.dataset = dataset
        self.args = args

    def __getitem__(self, item):
        return self.dataset[item]

    def __len__(self):
        return len(self.dataset)

    def config_make(self):
        return self.args
        pass


class DataSet_CRWU_MultiFile:
    """ 读取多个数据文件，划分数据集
        并自动配置成 dataset 的标准属性（通过Sub_Little_Dataset）
    """

    def __init__(self):
        pass

    def get_train_and_test_set_random(self, root, load_files: list = None, train_test_rate=0.7,
                                      repeat_win=1.0, window=2048, ):
        """ data.random_split 对每个类进行单独地随机划分数据集，各类样本是均衡的 """
        load_files0 = [
            '12k_Drive_End_B007_0_118.mat',
            '12k_Drive_End_B007_1_119.mat',
            '12k_Drive_End_B007_2_120.mat',
            '12k_Drive_End_B007_3_121.mat',
            '12k_Drive_End_B014_0_185.mat',
            '12k_Drive_End_B014_1_186.mat',
            # '12k_Drive_End_B014_2_187.mat',
        ]
        if load_files is None: load_files = load_files0
        num_class = len(load_files)
        train_dataset, test_dataset = [], []
        arg0 = None
        for idx, file in enumerate(load_files):
            dataset = DataSet_DIRG(root, [file], repeat_win, window, label=idx, num_class=num_class)
            arg0 = dataset.config_make()
            train_num = int(len(dataset) * train_test_rate)
            test_num = len(dataset) - train_num
            # print('train_num, test_num :', train_num, test_num, train_num + test_num, len(dataset))
            train_dataset0, test_dataset0 = data.random_split(dataset=dataset, lengths=[train_num, test_num], )
            train_dataset += train_dataset0
            test_dataset += test_dataset0

        ''' 配置 dataset 的类属性'''
        args = {
            'data path': str(root),
            'load files': load_files,
            'train_test_rate': train_test_rate,
            'repeat_win': repeat_win,
            'window': window,
            'log': 'All file\'s data were uniformly normalized! [by data.random_split: random]',
        }
        args = dict_add_items(arg0, args)
        train_dataset = Sub_Little_Dataset(train_dataset, args)
        test_dataset = Sub_Little_Dataset(test_dataset, args)
        return train_dataset, test_dataset

    def get_train_and_test_set_ordered(self, root, load_files: list = None, train_test_rate=0.7,
                                       repeat_win=1.0, window=2048, ):
        """ 按顺序划分数据集，各类样本是均衡的 """
        load_files0 = ['12k_Drive_End_B007_0_118.mat', ]
        if load_files is None: load_files = load_files0
        num_class = len(load_files)
        train_dataset, test_dataset = [], []
        arg0 = None
        for idx, file in enumerate(load_files):
            dataset = DataSet_DIRG(root, [file], repeat_win, window, label=idx, num_class=num_class)
            arg0 = dataset.config_make()
            train_num = int(len(dataset) * train_test_rate)
            test_num = len(dataset) - train_num
            # print('train_num, test_num :', train_num, test_num, train_num + test_num, len(dataset))
            # train_dataset0, test_dataset0 = data.random_split(dataset=dataset, lengths=[train_num, test_num], )  # 随机
            train_dataset0 = data.Subset(dataset, indices=list(np.arange(0, train_num)))  ## 根据索引indices提取子数据集
            test_dataset0 = data.Subset(dataset, indices=list(np.arange(train_num, train_num + test_num)))
            # print('\t', len(train_dataset0), len(test_dataset0))
            train_dataset += train_dataset0
            test_dataset += test_dataset0

        ''' 配置 dataset 的类属性'''
        args = {
            'data path': str(root),
            'load files': load_files,
            'train_test_rate': train_test_rate,
            'repeat_win': repeat_win,
            'window': window,
            'log': 'All file\'s data were uniformly normalized! [data.Subset: ordered]',
        }
        args = dict_add_items(arg0, args)
        train_dataset = Sub_Little_Dataset(train_dataset, args)
        test_dataset = Sub_Little_Dataset(test_dataset, args)
        return train_dataset, test_dataset

    def get_train_and_test_set_ordered_enhance(self, root, load_files: list = None, train_test_rate=0.7,
                                               repeat_win=1.0, window=2048, **kwargs):
        """ 按顺序划分数据集，各类样本是均衡的 """
        load_files0 = ['12k_Drive_End_B007_0_118.mat', ]
        if load_files is None: load_files = load_files0
        num_class = len(load_files)
        train_dataset, test_dataset = [], []
        arg0 = None
        for idx, file in enumerate(load_files):
            dataset = DataSet_DIRG_Signal_Process(root, [file], repeat_win, window, label=idx, num_class=num_class,
                                                  **kwargs)
            arg0 = dataset.config_make()
            train_num = int(len(dataset) * train_test_rate)
            test_num = len(dataset) - train_num
            # print('train_num, test_num :', train_num, test_num, train_num + test_num, len(dataset))
            # train_dataset0, test_dataset0 = data.random_split(dataset=dataset, lengths=[train_num, test_num], )  # 随机
            train_dataset0 = data.Subset(dataset, indices=list(np.arange(0, train_num)))  ## 根据索引indices提取子数据集
            test_dataset0 = data.Subset(dataset, indices=list(np.arange(train_num, train_num + test_num)))
            # print('\t', len(train_dataset0), len(test_dataset0))
            train_dataset += train_dataset0
            test_dataset += test_dataset0

        ''' 配置 dataset 的类属性'''
        args = {
            'data path': str(root),
            'load files': load_files,
            'train_test_rate': train_test_rate,
            'repeat_win': repeat_win,
            'window': window,
            'log': 'Data are enhance by random mask and noise! [data.Subset: ordered]',
        }
        args = dict_add_items(arg0, args)
        train_dataset = Sub_Little_Dataset(train_dataset, args)
        test_dataset = Sub_Little_Dataset(test_dataset, args)
        return train_dataset, test_dataset

    pass


""" 功能测试案例 """


def run():
    # import random
    # def seed_torch(seed=5):
    #     random.seed(seed)
    #     os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    #     torch.backends.cudnn.benchmark = False
    #     torch.backends.cudnn.deterministic = True
    # seed_torch()

    ''' 统计各类样本数量 '''

    def calculate_num(dataloader):
        labels = np.array([])
        for i, dat in enumerate(dataloader):
            de, label = dat
            label = label.reshape(-1, 1)
            if i == 0:
                labels = label
            else:
                labels = np.vstack([labels, label])
        labels = labels.reshape(-1)
        print(labels.shape)
        num_lab = {}
        for i in range(max(labels) + 1):
            num_lab[str(i)] = labels[labels == i].shape[0]
        return num_lab

    file = r'I:\python_datasets\CRWU\CRWU_original\12k_Drive_End_B007_1_119.mat'
    # dataset = DataSet_DIRG('I:\python_datasets\CRWU\CRWU_original', None, 2, 500)
    train_dataset, test_dataset = DataSet_CRWU_MultiFile().get_train_and_test_set_ordered(
        'I:\python_datasets\CRWU\CRWU_original', None,
        train_test_rate=0.7, repeat_win=0.6, window=2048)
    print(f'\ttrain_dataset len={len(train_dataset)}, test_dataset len={len(test_dataset)}')
    print(train_dataset.config_make())
    exit()
    trainloader = data.DataLoader(dataset=train_dataset, batch_size=512, shuffle=False, drop_last=False, )
    testloader = data.DataLoader(dataset=test_dataset, batch_size=512, shuffle=False, drop_last=False, )
    # print(trainloader.dataset.args)

    print('trainloader:', calculate_num(trainloader))
    print('testloader:', calculate_num(testloader))

    # from matplotlib import pyplot as plt
    #
    # for i in range(50):
    #     DE = train_dataset[i][:,:,0]
    #     y = signal_normalization.fft_transform(DE)
    #     plt.figure(1)
    #     plt.clf()
    #     plt.plot(DE)
    #     plt.draw()
    #     plt.pause(0.2)
    #     plt.figure(2)
    #     plt.clf()
    #     plt.plot(y)
    #     plt.draw()
    #     plt.pause(2)

    exit()
