import os
import pandas as pd
import time
import numpy as np
import pickle
import cv2 as cv
from matplotlib import pyplot as plt
import torch
import torch.nn as nn


# data_root = r'I:\python_datasets\ieee-phm-2012-data-challenge-dataset-master'
# file_path = os.path.join(data_root, 'Full_Test_Set', 'Bearing1_3', 'acc_00001.csv')
# print(file_path)
# # dataF = csv.reader(file_path)
# # dataF = pd.read_csv(file_path, header=None)
# # print(dataF.to_numpy())
# ts = '2010-1-1 {h}:{m}:{s}'.format(h='23', m='45', s='34')  # ,ms=dataF[3]


def data_file_format(file):
    ''' 读取振动数据 ，第一列为时间戳，第二列为水平振动，第三列为垂直振动'''
    data_all = None
    data_Fame = pd.read_csv(file, header=None)
    num = len(data_Fame)
    # print(data_Fame.shape)
    if data_Fame.shape[1] > 1:
        data_Fame = data_Fame.to_numpy()
        # print(data_Fame.shape)
        for i in range(data_Fame.shape[0]):
            dataF = data_Fame[i]
            # ts = '2010-1-1 {h}:{m}:{s}'.format(h=int(dataF[0]),m=int(dataF[1]),s=int(dataF[2])) #,ms=dataF[3]
            ts = '2010-1-1 %d:%d:%d' % (int(dataF[0]), int(dataF[1]), int(dataF[2]))  # ,ms
            tim = time.strptime(ts, '%Y-%m-%d %H:%M:%S')
            tim = float(time.mktime(tim)) + float('0.{ms}'.format(ms=int(dataF[3])))
            data_mat = np.array([tim], dtype=np.float).reshape(1, 1)
            data_mat = np.concatenate([data_mat, dataF[4:].reshape(1, -1)], axis=1)
            if data_all is None:
                data_all = data_mat
            else:
                data_all = np.concatenate([data_all, data_mat], axis=0)
    else:
        for dataF in data_Fame[0]:  ## range(len(data_Fame)):
            # dataF = str(data_Fame[i])
            dataF = dataF.split(';')
            ts = '2010-1-1 %d:%d:%d' % (int(dataF[0]), int(dataF[1]), int(dataF[2]))  # ,ms
            tim = time.strptime(ts, '%Y-%m-%d %H:%M:%S')
            tim = float(time.mktime(tim)) + float('0.{ms}'.format(ms=int(dataF[3])))
            tim = [tim] + dataF[4:]
            data_mat = np.array(tim, dtype=np.float).reshape(1, -1)
            # data_mat = np.concatenate([data_mat, dataF[4:].reshape(1, -1)], axis=1)
            if data_all is None:
                data_all = data_mat
            else:
                data_all = np.concatenate([data_all, data_mat], axis=0)

    # print( data_all.shape)
    if data_all.shape[0] != num:
        raise "[%s] data processing ERROR!" % file
    else:
        print('[%s] finish done! total num=%d' % (file, num))
    return data_all


# file_path = r'I:\python_datasets\ieee-phm-2012-data-challenge-dataset-master\Learning_set\Bearing1_2\temp_00089.csv'
# data_file_format(file_path)


class DataSet_IEEE_2012:
    ''' -----  数据集文件夹简介  ---------
        数据集一共有 3 种工况.
        训练集：[[Bearing1_1,Bearing1_2],[Bearing2_1,Bearing2_2],[Bearing3_1,Bearing3_2]]
        测试集：[[Bearing1_3,Bearing1_4,Bearing1_5,Bearing1_6,Bearing1_7],
                [Bearing2_3,Bearing2_4,Bearing2_5,Bearing2_6,Bearing2_7],
                [Bearing3_3]]
        测试集原数据：Full_Test_Set
    '''
    train_sets = ['Bearing1_1', 'Bearing1_2', 'Bearing2_1', 'Bearing2_2', 'Bearing3_1', 'Bearing3_2']
    test_sets = ['Bearing1_3', 'Bearing1_4', 'Bearing1_5', 'Bearing1_6', 'Bearing1_7',
                 'Bearing2_3', 'Bearing2_4', 'Bearing2_5', 'Bearing2_6', 'Bearing2_7',
                 'Bearing3_3']

    def __init__(self, root, data_type='', folder=None, data_content: list = None, repeat_win=1,
                 window=2048, rebuild=False):
        if data_content is None:
            data_content = ['vibration', 'temp']
        elif len(data_content) > 2:
            raise ('data_content can only be ["vibration", "temp"]')
        if data_type.lower() in ['train', 'test'] and folder is None:
            if data_type.lower() == 'train':
                folder = 'Learning_set'
                self.sub_folder = self.train_sets
                self.num_category = len(self.train_sets)
            elif data_type.lower() == 'test':
                folder = 'Test_set'
                self.sub_folder = self.test_sets
                self.num_category = len(self.test_sets)
        self.data_content = []
        for i, k in enumerate(data_content):
            if k.lower()[:4] == 'vibr': self.data_content.append('acc')
            if k.lower()[:4] == 'temp': self.data_content.append('temp')

        self.root = root  #os.path.join(root, folder)
        self.repeat_win = repeat_win
        self.window = window
        self.len_size = None
        ## 制作缓存文件
        cached_file = os.path.join(self.root, 'processed_cached.pkl')
        self.data_file = cached_file
        if os.path.exists(cached_file) and not rebuild:
            print('Cached file exist! %s' % cached_file)
            pass
        else:
            data = self.preprocessing_data_in_folder()
            with open(cached_file, "wb") as f:
                pickle.dump(data, f)
            print("Save Cached file: {}".format(cached_file))
            self.preprocessing_data_normalization(time_norm=True, vibr_norm=True, temp_norm=False, filename=cached_file)
        self.data_list = self.load_cached(False)
        pass

    def preprocessing_data_in_folder(self):
        folder_data_dict = {}
        for sub_folder in self.sub_folder:
            sub_path = os.path.join(self.root, str(sub_folder))
            vibr_data = None
            temp_data = None
            data_files_list = os.listdir(sub_path)
            for file in data_files_list:
                if file.startswith('acc'):
                    data = data_file_format(os.path.join(sub_path, file))
                    if vibr_data is None:
                        vibr_data = data
                    else:
                        vibr_data = np.concatenate([vibr_data, data], axis=0)
                if file.startswith('temp'):
                    data = data_file_format(os.path.join(sub_path, file))
                    if temp_data is None:
                        temp_data = data
                    else:
                        temp_data = np.concatenate([temp_data, data], axis=0)
            v = {sub_folder: {'vibr': vibr_data, 'temp': temp_data}}
            folder_data_dict = dict(folder_data_dict, **v)
        return folder_data_dict

    def preprocessing_data_normalization(self, time_norm=True, vibr_norm=False, temp_norm=False, filename=None):
        if time_norm | vibr_norm | temp_norm != True: return 0  ## 至少有一个需求
        with open(self.data_file, "rb") as f:
            data = pickle.load(f)
        for cate in data.keys():  ## 多工况下的多故障
            i_vibr, i_temp = data[cate]['vibr'], data[cate]['temp']
            if i_vibr is not None:
                if time_norm:
                    min_t, max_t = i_vibr[:, 0].min(), i_vibr[:, 0].max()
                    i_vibr[:, 0] = (i_vibr[:, 0] - min_t) / (max_t - min_t)
                if vibr_norm:
                    Ev = np.mean(i_vibr[:, 1:], axis=0)
                    Sv = np.var(i_vibr[:, 1:], axis=0)
                    i_vibr[:, 1:] = (i_vibr[:, 1:] - Ev) / Sv
            if i_temp is not None:
                if time_norm:
                    min_t, max_t = i_temp[:, 0].min(), i_temp[:, 0].max()
                    i_temp[:, 0] = (i_temp[:, 0] - min_t) / (max_t - min_t)
                if temp_norm:  ## 温度曲线近似递增，不适合归一化
                    # min_t, max_t = i_temp[:, 1:].min(axis=0), i_temp[:, 1:].max(axis=0)
                    # i_temp[:, 1:] = (i_temp[:, 1:] - min_t) / (max_t - min_t)
                    pass

            data[cate]['vibr'], data[cate]['temp'] = i_vibr, i_temp
        if filename is None: filename = self.data_file
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print("Save Cached normalization file: {}".format(self.data_file))

    def __len__(self):
        if self.len_size is not None: return self.len_size
        data_list = self.load_cached(False)
        self.data_list = data_list
        size_arr = None
        for i_cate in range(len(data_list)):
            [i_vibr, i_temp] = data_list[i_cate]
            l1, l2 = 0, 0
            if i_vibr is not None:
                l1 = i_vibr.shape[0]
            if i_temp is not None:
                l2 = i_temp.shape[0]
            s = np.array([l1, l2]).reshape(1, 2)
            if size_arr is None:
                size_arr = s
            else:
                size_arr = np.concatenate([size_arr, s], axis=0)
        # print(size_arr)
        size = size_arr.max(axis=0)
        # print(size)
        self.len_size = int(size[0] / self.window) * self.num_category  #item 每次读取一个类
        return self.len_size
        pass

    def __getitem__(self, item):
        idx = int(item/self.num_category)
        idx2 = int(item%self.num_category)
        data_list = self.data_list  # self.load_cached(ishow=False)
        data_list = self.get_Onewindow_all_category(data_list, idx, self.repeat_win, 0)
        label, vibr_h, vibr_v, temp = self.get_Onebatch_one_category_normalization(data_list, idx2)
        label = torch.tensor(1-label, dtype=torch.float)
        vibr_h = torch.tensor(vibr_h, dtype=torch.float)
        vibr_v = torch.tensor(vibr_v, dtype=torch.float)
        if temp is not None: temp = torch.tensor(temp, dtype=torch.float)
        # else:temp = torch.tensor([])
        # if temp is not None:
        #     print(item, '===', label.shape, vibr_h.shape, vibr_v.shape, temp.shape)
        # else:
        #     print(item, '===', label.shape, vibr_h.shape, vibr_v.shape)
        return label, vibr_h, vibr_v, temp
        pass

    def collate_fn(self,data):
        res = []
        for d in data:
            # label, vibr_h, vibr_v, temp = d
            for i,sub_d in enumerate(d):
                if len(res)<=i:   res.append([])
                res[i].append(sub_d)
        res2 = []
        for i, sub_d in enumerate(res):
            try:
                d = torch.stack(sub_d, dim=0)
            except Exception as e:
                d = sub_d
                # print('\tcollate_fn:  %d is list' %i, e)
            res2.append(d)
        return res2


    def load_cached(self, ishow=True):
        with open(self.data_file, "rb") as f:
            data = pickle.load(f)
        resdata = []
        for cate in data.keys():  ## 多工况下的多故障
            i_vibr, i_temp = data[cate]['vibr'], data[cate]['temp']
            if ishow:
                if i_vibr is None or i_temp is None:
                    print(cate, '\tdata size=', i_vibr.shape)
                    print('\t\t', i_vibr.min(axis=0), i_vibr.max(axis=0))
                else:
                    print(cate, '\tdata size=', i_vibr.shape, i_temp.shape)
                    print('\t\t', i_vibr.min(axis=0), i_vibr.max(axis=0), i_temp.min(axis=0), i_temp.max(axis=0))
            resdata.append([i_vibr, i_temp])
        return resdata
        pass

    def get_Onewindow_all_category(self, data_list: list, idx=0, repeat_win=1, window=0):
        ''' 以vibr数据为基准（vibr比temp采样密度更高），从time_idx处往后截取一个window数据，获取末尾时间time_end
            然后，在temp中截取相同时间段的数据T。此时，T的维度不一定等于一个window。'''

        def get_window_data_autorepeat_vibr(dat, start, end):
            if dat is None: return None
            start0, end = int(start), int(end)
            w, d = dat.shape[0], dat.shape[1]
            win = end - start0
            start = start0 - int(start0 / w) * w  ## 取余
            end = start + win
            global out_vibr
            if end > w:
                if start < w -int( 3 / 4 * win):  ## 3/4有效，后面用0补齐
                    out_vibr = dat[start:, :].copy()
                    out_vibr.resize([win,d])
                    out_vibr[-1, 0] = 1.0
                    #return out_vibr
                else:  ## 尾数不足，倒退补满
                    start = w-win
                    end = w
                    out_vibr = dat[start:end]
                    out_vibr = out_vibr.copy()
                    out_vibr.resize([win, d])
                    out_vibr[-1, 0] = 1.0
            else:
                out_vibr = dat[start:end]
            # if out_vibr is None:
            #     print('get_window_data_autorepeat_vibr', start0, start, w, end)

            return out_vibr

        def get_window_data_autorepeat_temp(dat, start, end):
            if dat is None: return None
            start, end = float(start), float(end)
            out_temp = dat[start <= dat[:, 0]]
            out_temp = out_temp[out_temp[:, 0] <= end]
            return out_temp

        if window == 0: window = self.window
        start = idx  # + 1 - repeat_win
        # start = start if start >= 0 else 0
        start, end = start * window, (start + repeat_win) * window
        out_data = list()
        for i_cate in range(len(data_list)):
            [i_vibr, i_temp] = data_list[i_cate]
            # print('====', i_vibr.shape,  start, end)
            out_vibr = get_window_data_autorepeat_vibr(i_vibr, start, end)
            t1, t2 = out_vibr[0, 0], out_vibr[-1, 0]
            out_temp = get_window_data_autorepeat_temp(i_temp, t1, t2)
            # print('====', i_temp.shape , out_temp.shape,  t1, t2 , out_temp)
            out_data.append([out_vibr, out_temp])
        return out_data

    def config_make(self):
        args = {
            'root path': str(self.root),
            'log': 'All file\'s data were uniformly normalized!',
        }
        return args

    def get_Onebatch_one_category_normalization(self, data_list: list, idx=0):
        ''' 从提取的窗口data中，再提取具体的种类'''
        data_vibr, data_temp = data_list[idx]
        label = data_vibr[:, 0].mean().reshape([1])
        vibr_h = data_vibr[:, 1].reshape(1, -1)
        vibr_v = data_vibr[:, 2].reshape(1, -1)
        temp = None
        try:
            if data_temp is not None:
                # print('===', data_temp.shape)
                temp = data_temp[:, 1]
                w = vibr_h.shape[1]
                temp = cv.resize(temp, (1, w), interpolation=cv.INTER_LINEAR)
                temp = np.array(temp).reshape(1, -1)
            if temp.shape[0]!=1: temp=None
        except:pass
        return label, vibr_h, vibr_v, temp

    def plot_vibr(self, draw_data):
        """ 绘制振动信号图 """
        plt.figure(dpi=600)  # figsize=(416, 416),
        from scipy import signal
        Pxx, freqs, bins, im = plt.specgram(draw_data, NFFT=512, Fs=20000000, noverlap=126,
                                            window=signal.get_window(('kaiser', 18.0), 512))

        # plt.xlim((0, 0.001))
        # plt.ylim((-10000000, 10000000))
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        # 取消显示坐标轴
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.show()
        # plt.savefig('./example1.jpg', pad_inches=0)

    def plot_vibr2(self, draw_data):
        plt.figure(dpi=300)
        plt.plot(draw_data)
        plt.show()


if __name__ == '__main__':
    root = r'I:\python_datasets\ieee-phm-2012-data-challenge-dataset-master'
    dataset = DataSet_IEEE_2012(root, 'train', )
    num = len(dataset)
    for i in range(num):
        ii = num - i
        label, vibr_h, vibr_v, temp = dataset[i]
        if temp is not None:
            print(i, '===', label.shape, vibr_h.shape, vibr_v.shape, temp.shape)
        else:
            print(i, '===', label.shape, vibr_h.shape, vibr_v.shape)


    # dataset.plot_vibr2(dat[1][:, 1])
