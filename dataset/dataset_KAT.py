import os
import pickle

import scipy.io as scio
import pandas as pd
import numpy as np
import torch
import torch.nn.functional
from torch.utils import data
from utils_tool import signal_normalization

""" 德国-帕德博恩大学轴承数据集 KAT PU """


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


class DataSet_KAT(data.Dataset):
    """ -----  数据集文件夹简介  ---------
        文件使用官方数据集（文件名末尾带编号）
        每个 .mat 文件，有info、x、y和Description四个数据集，其中所有数据均存储在Y这个结构体下
        Y结构体包含了电机的电流数据（2个，大小为1X256823），温度数据（1个，大小1X4），机械参数（负载（force）、转速（speed）、扭矩（torque））和振动数据（1个，大小为1X256823）
        电机电流的采样频率为：64KHz
        振动信号的采样频率为：64KHz
        机械参数(加载力，加载力矩，速度)的采样频率:4KHz
        温度的采样频率为：1Hz
        一个mat文件采集时间为4s，每次实验进行20次左右
    """
    """ 统一读取所有文件后使用 data.random_split 划分训练集和测试集，会导致各类的数量不均，
        应该独立对每个类使用data.random_split划分数据集 
    """

    def __init__(self, root, load_files=None, repeat_win=1.0, window=2048, label: int = None, num_class: int = None,
                 train_test_rate: float = 0.7, cate: str = None, down_fre=12, ):
        super(DataSet_KAT, self).__init__()
        # root = r'I:\python_datasets\PU dataset'  # root  # os.path.join(root, folder)
        category = ['vibration']  ##, 'current'
        load_files0 = [
            [r'KA01\N09_M07_F10_KA01_3.mat', r'KA01\N09_M07_F10_KA01_5.mat',],
            [r'KA01\N09_M07_F10_KA01_12.mat', r'KA01\N09_M07_F10_KA01_15.mat',],
        ]
        if load_files is None: load_files = load_files0
        # files = [os.path.join(root, f) for f in load_files]
        self.all_data_arr_DE_FE_BA_label = self.load_files_and_format(root, load_files, win=window,
                                                                      repeat_win=repeat_win,
                                                                      cut_same=True, category=category,
                                                                      down_fre=down_fre,  # down_fre
                                                                      label=label, num_class=num_class,
                                                                      ishow=False)
        # """ 数据集内容变更"""
        if cate is not None:
            length = len(self.all_data_arr_DE_FE_BA_label[1])
            depart = int(length * train_test_rate)
            if cate.lower().startswith('train'):
                self.all_data_arr_DE_FE_BA_label = [self.all_data_arr_DE_FE_BA_label[0][:depart],
                                                    self.all_data_arr_DE_FE_BA_label[1][:depart]]
            else:
                self.all_data_arr_DE_FE_BA_label = [self.all_data_arr_DE_FE_BA_label[0][depart:],
                                                    self.all_data_arr_DE_FE_BA_label[1][depart:]]

        self.args = {
            'program name': 'DataSet_KAT',
            'program path': os.path.abspath(__file__),
            'data path': str(root),
            'load files': str(load_files),
            'repeat_win': repeat_win,
            'window': window,
            'num_class': num_class,
            'log': 'All file\'s data were uniformly normalized!',
        }

        # self.idx = np.arange(0, self.all_data_arr_DE_FE_BA_label[1].shape[0])
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

    def load_file_onlyone(self, path: str, win: int, repeat_win: float, cut_same=False, cate: list = None,
                          down_fre=None, ishow=False):
        if cate is None: cate = self.category
        """ 官方数据中有存在放置错误的情况 """
        filename = os.path.basename(path)[:-4]
        # print(filename)
        data = scio.loadmat(path)[filename]['Y'][0, 0][0]
        num_record = {'origal': {}, 'new': {}}
        tmplist = []  # *len(cate)  # 0:vibration  1:current   2:current
        for idx in range(len(data['Name'])):
            name = data['Name'][idx]
            is_load = False
            for n in list(cate):
                if n in str(name): is_load = True
            if is_load:
                tmp = data['Data'][idx]
                tmp = np.squeeze(tmp)
                if ishow: print('\t\tdat:', idx, '---', tmp.shape)
                if 'vibration_1' == name:
                    if down_fre is not None:
                        tmp = sample_frequency_change(tmp, ori_sample_frequency=64, tar_sample_frequency=12)
                    tmp_chunk = seq_chunk(tmp, chunk=win, step=int(win * (1 - repeat_win)),
                                          log=path + '  ' + str(name) + '   ' + str(idx),
                                          ishow=ishow)
                    for i in range(len(tmplist), 1): tmplist.append([])
                    tmplist[0] = tmp_chunk
                    num_record['origal']['vibration'] = tmp.shape[0]
                    num_record['new']['vibration'] = tmp_chunk.shape[0]
                elif 'phase_current_1' == name:
                    if down_fre is not None:
                        tmp = sample_frequency_change(tmp, ori_sample_frequency=64, tar_sample_frequency=down_fre)
                    tmp_chunk = seq_chunk(tmp, chunk=win, step=int(win * (1 - repeat_win)),
                                          log=path + '  ' + str(name) + '   ' + str(idx),
                                          ishow=ishow)
                    for i in range(len(tmplist), 2): tmplist.append([])
                    tmplist[1] = tmp_chunk
                    num_record['origal']['phase_current_1'] = tmp.shape[0]
                    num_record['new']['phase_current_1'] = tmp_chunk.shape[0]
                elif 'phase_current_2' == name:
                    if down_fre is not None:
                        tmp = sample_frequency_change(tmp, ori_sample_frequency=64, tar_sample_frequency=down_fre)
                    tmp_chunk = seq_chunk(tmp, chunk=win, step=int(win * (1 - repeat_win)),
                                          log=path + '  ' + str(name) + '   ' + str(idx),
                                          ishow=ishow)
                    for i in range(len(tmplist), 3): tmplist.append([])
                    tmplist[2] = tmp_chunk
                    num_record['origal']['phase_current_2'] = tmp.shape[0]
                    num_record['new']['phase_current_2'] = tmp_chunk.shape[0]
                elif 'speed' == name:
                    if down_fre is not None:
                        tmp = sample_frequency_change(tmp, ori_sample_frequency=4, tar_sample_frequency=down_fre)
                    tmp_chunk = seq_chunk(tmp, chunk=win, step=int(win * (1 - repeat_win)),
                                          log=path + '  ' + str(name) + '   ' + str(idx),
                                          ishow=ishow)
                    for i in range(len(tmplist), 4): tmplist.append([])
                    tmplist.append([])
                    tmplist[3] = tmp_chunk
                    num_record['origal']['speed'] = tmp.shape[0]
                    num_record['new']['speed'] = tmp_chunk.shape[0]
        # print( 'load_file_onlyone:', num_record['new'].values() )
        if len(num_record['new'].values()) < 1:
            print(num_record, path, '\n\t\t', data['Name'])
        if cut_same:
            minnum = np.array(list(num_record['new'].values())).min(axis=0)
            for i in range(len(tmplist)):
                if tmplist[i].shape[0] > minnum:
                    tmplist[i] = tmplist[i][:minnum, :]
        # if len(tmplist) < 3: tmplist.append([])
        return tmplist, num_record

    ''' 加载所有文件（files），缺失列用0补齐 '''

    def load_files_and_format(self, path, files: [], win: int, repeat_win: float, cut_same=False, category=None,
                              down_fre=None, label: int = None, num_class: int = None, ishow=False):
        # files =[ [], []]
        all_data_arr_DE_FE_BA = np.array([])
        all_data_label = np.array([])
        if num_class is None: num_class = len(files)
        for idx, file_list in enumerate(files):
            for idx2, sub_file in enumerate(file_list):
                file = os.path.join(path, sub_file)
                datalist, numdict = self.load_file_onlyone(file, win, repeat_win, False, category, down_fre,
                                                           ishow=ishow)  ## 为了拼接，cut_same=true,但，减少重复工作

                minnum = np.array(list(numdict['new'].values())).min(axis=0)
                # if len(numdict['new'].keys()) <= 2:  ## 补齐缺失的一类传感器数据
                #     datalist[2] = np.zeros_like(datalist[0])
                if label is None:
                    label2 = idx
                else:
                    label2 = label

                label_arr = (np.ones(shape=[minnum, 1], dtype=np.int) * label2)  ## 创建该文件对应的 label
                # label_arr = (np.zeros(shape=[minnum, num_class], dtype=np.int)) ## 分类问题需要独热编码
                # label_arr[:,label2] = 1 ## 分类问题需要独热编码

                if not cut_same: raise ValueError("all data should be the same length! [cut_same=true]")
                tmp_stack = []
                for i in range(len(datalist)): tmp_stack.append(datalist[i][:minnum])
                datalist = np.stack(tmp_stack, axis=2)

                if all_data_arr_DE_FE_BA.any():
                    all_data_arr_DE_FE_BA = np.vstack([all_data_arr_DE_FE_BA, datalist])
                    all_data_label = np.vstack([all_data_label, label_arr])
                else:
                    all_data_arr_DE_FE_BA = datalist
                    all_data_label = label_arr
                # print('data shape log:', all_data_label[-1], all_data_arr_DE_FE_BA.shape, all_data_label.shape, all_data_arr_DE_FE_BA.shape[0] - datalist.shape[0])

        # all_data_label = torch.nn.functional.one_hot(all_data_label) ## 分类问题需要独热编码
        return all_data_arr_DE_FE_BA, all_data_label

    def __getitem__(self, item):
        """ 使用 data.random_split 后，这里报错不会有提醒 """
        dat, label = self.all_data_arr_DE_FE_BA_label[0], self.all_data_arr_DE_FE_BA_label[1]
        vibr,  = dat[item, :, 0]
        label = label[item][0]
        # print(DE.shape, FE.shape, BA.shape, label.shape)  ## (1000,) (1000,) (1000,) (241,)
        """ signal processing """
        # DE = signal_normalization.add_gauss_noise(DE, 1)

        vibr = torch.tensor(vibr, dtype=torch.float).reshape(1, -1)

        label = torch.tensor(label, dtype=torch.long)  # .reshape(1)
        return vibr, vibr, vibr, label  # self.idx[item]
        # return DE, FE, BA, label  # self.idx[item]
        pass

    def __len__(self):
        return self.all_data_arr_DE_FE_BA_label[1].shape[0]

    def config_make(self):
        return self.args
        pass

    # def collate_fn(self, data):
    #     return data


class DataSet_KAT_Signal_Process(DataSet_KAT):
    """ -----  数据集文件夹简介  ---------
            文件使用官方数据集（文件名末尾带编号）
            每个 .mat 文件，有info、x、y和Description四个数据集，其中所有数据均存储在Y这个结构体下
            Y结构体包含了电机的电流数据（2个，大小为1X256823），温度数据（1个，大小1X4），机械参数（负载（force）、转速（speed）、扭矩（torque））和振动数据（1个，大小为1X256823）
            电机电流的采样频率为：64KHz
            振动信号的采样频率为：64KHz
            机械参数(加载力，加载力矩，速度)的采样频率:4KHz
            温度的采样频率为：1Hz
            一个mat文件采集时间为4s，每次实验进行20次左右
        """
    """ 
        数据提取时，加入信号处理操作
    """

    def __init__(self, root, load_files=None, repeat_win=1.0, window=2048, label: int = None, num_class: int = None,
                 train_test_rate: float = 0.7, cate: str = None, down_fre=12, mask_win: int = None, mask_start_idx: int = None,
                 snr: float = None):
        super(DataSet_KAT_Signal_Process, self).__init__(root, load_files, repeat_win, window, label, num_class,
                                                         train_test_rate, cate, down_fre)

        self.mask_win = mask_win
        self.mask_start_idx = mask_start_idx
        self.snr = snr
        self.args['program name'] = 'DataSet_KAT_Signal_Process'
        mask_win = round(mask_win/window, 1) if mask_win is not None else 'random'
        mask_start_idx = round(mask_start_idx / window, 1) if mask_start_idx is not None else 'random'
        snr = round(snr / window, 1) if snr is not None else 'random'
        self.args['signal enhance'] = dict(mask_win=mask_win, mask_start_idx=mask_start_idx, SNR=snr)
        # print('DataSet_CRWU_Signal_Process:', self.mask_win , self.mask_start_idx  )

    def __getitem__(self, item):
        """ 使用 data.random_split 后，这里报错不会有提醒 """
        dat, label = self.all_data_arr_DE_FE_BA_label[0], self.all_data_arr_DE_FE_BA_label[1]
        vibr = dat[item, :, 0]
        label = label[item][0]

        """ signal processing """  # Fixme
        vibr, real_rate, start_rate = self.signal_processing(vibr, np.random.randint(0, 3))
        # if real_rate<0.5:
        #     label = 10
        # print('====real_rate:', real_rate, label)
        vibr = torch.tensor(vibr, dtype=torch.float).reshape(1, -1)
        label = torch.tensor(label, dtype=torch.long)  # .reshape(1)
        rate = torch.tensor(real_rate, dtype=torch.float).reshape(1)  #
        return vibr, vibr, vibr, label, rate, start_rate  # self.idx[item]
        # return DE, FE, BA, label, rate, start_rate  # self.idx[item]
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


class DataSet_KAT_MultiFile:
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
            dataset = DataSet_KAT(root, [file], repeat_win, window, label=idx, num_class=num_class)
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
            dataset = DataSet_KAT(root, [file], repeat_win, window, label=idx, num_class=num_class)
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
            dataset = DataSet_KAT_Signal_Process(root, [file], repeat_win, window, label=idx, num_class=num_class, **kwargs)
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


