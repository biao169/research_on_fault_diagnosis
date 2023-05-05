import os
import pickle

import scipy.io as scio
import pandas as pd
import numpy as np
import torch
import torch.nn.functional
from torch.utils import data
from utils_tool import signal_normalization

""" 数据集来自都灵理工大学的轴承数据MFPT """


def sample_frequency_change(seq, ori_sample_frequency=64, tar_sample_frequency=12, ):
    """ 采样频率变更 [ sample_frequency/kHz ]"""
    if tar_sample_frequency >= ori_sample_frequency: return seq
    seq_length = seq.shape[0]
    step = ori_sample_frequency // tar_sample_frequency
    idx = np.arange(0, seq_length, step)
    seq = seq[idx]
    return seq


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


class DataSet_MFPT(data.Dataset):
    """ -----  数据集文件夹简介  ---------
        文件使用官方数据集（文件夹为类别）
    """
    """ 统一读取所有文件后使用 data.random_split 划分训练集和测试集，会导致各类的数量不均，
        应该独立对每个类使用data.random_split划分数据集 
    """

    def __init__(self, root, load_files=None, repeat_win=1.0, window=2048, label: int = None, num_class: int = None,
                 train_test_rate: float = 0.7, cate: str = None):
        super(DataSet_MFPT, self).__init__()
        # root = root  # os.path.join(root, folder)
        load_files0 = [
            ['1 - Three Baseline Conditions\\baseline_1.mat', 97656],
            ['2 - Three Outer Race Fault Conditions\\OuterRaceFault_1.mat', 48828]
        ]
        if load_files is None: load_files = load_files0
        if num_class is None: num_class = len(load_files)
        # datalist, numdict = self.load_file_onlyone(files[0], window, repeat_win, False,
        #                                            ishow=True)  ## 为了拼接，cut_same=true,但，减少重复工作
        # exit()

        self.all_data_arr_x_y_z_label = self.load_files_and_format(root, load_files, win=window, repeat_win=repeat_win,
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
            'program name': 'DataSet_MFPT',
            'program path': os.path.abspath(__file__),
            'data path': str(root),
            'load files': str(load_files),
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

    def load_file_onlyone(self, path: str, win: int, repeat_win: float, ori_frequencyofre: int, tar_frequencyofre=48828,
                          ishow=False):
        """ 官方数据中有存在放置错误的情况 """

        data = scio.loadmat(path)
        data = data['bearing'][0][0]['gs']
        tmp = np.squeeze(data, axis=1)
        # print( tmp.shape)
        # tmp = np.squeeze(tmp)
        tmp = sample_frequency_change(tmp, ori_sample_frequency=ori_frequencyofre,
                                      tar_sample_frequency=tar_frequencyofre)

        """ 特殊情况 """
        if ori_frequencyofre == 97656: tmp = tmp[:146484]

        tmp_chunk = seq_chunk(tmp, chunk=win, step=int(win * (1 - repeat_win)),
                              log=path + '  ',
                              ishow=ishow)

        return tmp_chunk

    ''' 加载所有文件（files），缺失列用0补齐 '''

    def load_files_and_format(self, root, files: [], win: int, repeat_win: float, cut_same=False,
                              label: int = None, num_class: int = None, ishow=False):
        all_data_arr_DE_FE_BA = np.array([])
        all_data_label = np.array([])
        for idx, file_msg in enumerate(files):
            file = os.path.join(root, file_msg[0])
            datalist = self.load_file_onlyone(file, win, repeat_win,
                                              ori_frequencyofre=int(file_msg[1]), tar_frequencyofre=48828,
                                              ishow=ishow)  ## 为了拼接，cut_same=true,但，减少重复工作
            # print( 'file:', file , file_msg)
            minnum = datalist.shape[0]
            if label is None:
                label2 = idx
            else:
                label2 = label

            label_arr = (np.ones(shape=[minnum, 1], dtype=np.int) * label2)  ## 创建该文件对应的 label
            # label_arr = (np.zeros(shape=[minnum, num_class], dtype=np.int)) ## 分类问题需要独热编码
            # label_arr[:,label2] = 1 ## 分类问题需要独热编码

            if not cut_same: raise ValueError("all data should be the same length! [cut_same=true]")

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
        vibr = dat[item, :]
        label = label[item, 0]
        # print( vibr.shape , label.shape)
        """ signal processing """
        # DE = signal_normalization.add_gauss_noise(DE, 1)

        dat1 = torch.tensor(vibr, dtype=torch.float).reshape(1, -1)
        label = torch.tensor(label, dtype=torch.long)  # .reshape(1)
        return label, torch.tensor([1], dtype=torch.float), torch.tensor([1], dtype=torch.float), dat1  # self.idx[item]
        pass

    def __len__(self):
        return self.all_data_arr_x_y_z_label[1].shape[0]

    def config_make(self):
        return self.args
        pass

    # def collate_fn(self, data):
    #     return data


class DataSet_MFPT_Signal_Process(DataSet_MFPT):
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
        super(DataSet_MFPT_Signal_Process, self).__init__(root, load_files, repeat_win, window, label, num_class,
                                                          train_test_rate, cate)

        self.mask_win = mask_win
        self.mask_start_idx = mask_start_idx
        self.snr = snr
        self.args['program name'] = 'DataSet_MFPT_Signal_Process'
        mask_win = round(mask_win / window, 1) if mask_win is not None else 'random'
        mask_start_idx = round(mask_start_idx / window, 1) if mask_start_idx is not None else 'random'
        snr = round(snr / window, 1) if snr is not None else 'random'
        self.args['signal enhance'] = dict(mask_win=mask_win, mask_start_idx=mask_start_idx, SNR=snr)
        # print('DataSet_MFPT_Signal_Process:', self.mask_win , self.mask_start_idx  )

    def __getitem__(self, item):
        """ 使用 data.random_split 后，这里报错不会有提醒 """

        dat, label = self.all_data_arr_x_y_z_label[0], self.all_data_arr_x_y_z_label[1]
        vibr = dat[item, :]
        label = label[item][0]
        # print(vibr.shape, label.shape)  ## (1000,) (1000,) (1000,) (241,)
        """ signal processing """  # Fixme
        vibr, real_rate, start_rate = self.signal_processing(vibr, np.random.randint(0, 3))
        # if real_rate<0.5:
        #     label = 10
        # print('====real_rate:', real_rate, label)
        dat1 = torch.tensor(vibr, dtype=torch.float).reshape(1, -1)

        label = torch.tensor(label, dtype=torch.long)  # .reshape(1)
        rate = torch.tensor(real_rate, dtype=torch.float).reshape(1)  #
        return label, rate, start_rate, dat1  # self.idx[item]
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


class DataSet_MFPT_MultiFile:
    """ 读取多个数据文件，划分数据集
        并自动配置成 dataset 的标准属性（通过Sub_Little_Dataset）
    """

    def __init__(self):
        pass

    def get_train_and_test_set_random(self, root, load_files: list = None, train_test_rate=0.7,
                                      repeat_win=1.0, window=2048, ):
        """ data.random_split 对每个类进行单独地随机划分数据集，各类样本是均衡的 """
        num_class = len(load_files)
        train_dataset, test_dataset = [], []
        arg0 = None
        for idx, file in enumerate(load_files):
            dataset = DataSet_MFPT(root, [file], repeat_win, window, label=idx, num_class=num_class)
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
        num_class = len(load_files)
        train_dataset, test_dataset = [], []
        arg0 = None
        for idx, file in enumerate(load_files):
            dataset = DataSet_MFPT(root, [file], repeat_win, window, label=idx, num_class=num_class)
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
        num_class = len(load_files)
        train_dataset, test_dataset = [], []
        arg0 = None
        for idx, file in enumerate(load_files):
            dataset = DataSet_MFPT_Signal_Process(root, [file], repeat_win, window, label=idx, num_class=num_class,
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
