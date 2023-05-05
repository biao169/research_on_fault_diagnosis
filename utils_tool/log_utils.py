import os
import pandas as pd
import numpy as np
from tensorboardX import SummaryWriter
import yaml
import threading
# coding = utf-8
import configparser


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    warning = '\033[1;33;46m'  ## 高亮显示-黄色-白色
    error = '\033[1;31m'  ## 高亮显示-红色-默认
    tip = '\033[0;32m'  ## 默认-绿色-默认


class YamlHandler:
    """ Read or Save config file """

    def __init__(self, file):
        self.file = file

    def read_yaml(self, encoding='utf-8'):
        """读取yaml数据"""
        with open(self.file, encoding=encoding) as f:
            return yaml.load(f.read(), Loader=yaml.FullLoader)

    def write_yaml(self, data, encoding='utf-8'):
        """向yaml文件写入数据"""
        with open(self.file, encoding=encoding, mode='w') as f:
            return yaml.dump(data, stream=f, allow_unicode=True)


class Summary_Log:
    """
        >> 训练日志保存功能程序  <<
        程序根据 add_scalars 传入训练数据， 可选同时保存在csv文件中！
        启动 tensorboard 的方法是：在终端上执行命令： tensorboard --logdir={path}
    """

    def __init__(self, path, headers: dict, write_csv=False, tm_str='', save_log=True,
                 tensorboard_mode='train-test', new_thread=True, **kwargs):
        """
        :param  save_log: 控制是否保存训练日志[tensor & config]。如果不保存，也会创建相应的文件夹，同时有打印输出
        """
        if headers is None:
            headers = {'train': ['loss'], 'test': ['loss'], 'log': ['lr', 'time']}
        self.headers = headers
        self.separator = ' '  # 文件内，表头的分隔符
        csv_head = []
        csv_head.append('idx')
        for father in headers.keys():
            for sub_head in headers[father]:
                name = father + self.separator + sub_head
                csv_head.append(name)
        self.csv_head = csv_head
        self.data_buff = {name: [] for name in self.csv_head}
        self.newThread = new_thread
        """ 创建日志文件夹 """
        path_log = os.path.join(path, 'log')
        os.makedirs(path_log, exist_ok=True)

        self.save_log = save_log
        if not save_log:
            print(bcolors.warning + '[[log]:\tThe training process data [ tensor & config ] will not be stored!' + bcolors.ENDC, flush=True)
        self.config_file = os.path.join(path, 'config {}.yaml'.format(tm_str))
        """ csv文件控制项 """
        self.csv_log_file = os.path.join(path_log, 'train data {}.xlsx'.format(tm_str))
        self.write_csv = write_csv

        """ tensorboard 控制项 """
        if tensorboard_mode not in ['train-test', 'loss-acc']:
            raise ValueError(f'tensorboard_mode [{tensorboard_mode}] should be one of ["train-test", "loss-acc"]')
        self.tensorboard_mode = tensorboard_mode
        self.logWriter = None  ## SummaryWriter(path2, comment='log', )
        # path_tensor = os.path.join(path_log, 'tensor')
        # self.tensor_log_file = self.update_auto_pathname_for_tensor(path_tensor)

    def make_tensor_path(self):
        path_log = os.path.dirname( self.csv_log_file)
        path_tensor = os.path.join(path_log, 'tensor')
        self.tensor_log_file = self.update_auto_pathname_for_tensor(path_tensor)
        path = self.tensor_log_file
        self.prepare_tensor_path(path)

    def add_scalars(self, main_tag: str, scalars: {}, step: int, tolerant=False):
        """ 另用线程，减少占用训练的时间 """
        if self.newThread:
            t = threading.Thread(target=self.add_scalars_, args=(main_tag, scalars, step, tolerant))
            t.start()
        else:
            self.add_scalars_(main_tag, scalars, step, tolerant)

    def update_auto_pathname_for_tensor(self, path):
        """
            tensorboard 的保存文件夹名称更新
            由于多个tensorboard保存在一个文件夹下会出现读取混乱，故，需要重新命名
        """
        try:
            os.removedirs(path)  ## 现有文件夹是空文件夹，已移除
        except FileNotFoundError:
            pass
        except:  # 非空文件夹，重新改名
            pass
        tm_str = os.path.basename(self.config_file)[:-5]
        tm_str = tm_str[len('config '):]  ## 提取时间戳
        father_path = os.path.dirname(path)
        name = os.path.basename(path)
        file_list = os.listdir(father_path)
        num = 0
        while True:  ## 最多可重复做这么多次实验
            num += 1
            if num == 1:
                name2 = name
            else:
                name2 = name + '%02d' % num
            # if name2 not in file_list or num > 1000:  ## tensor 文件夹不带时间戳时可用
            #     break
            isout = True
            for f in file_list:
                if str(f).split(' ')[0] == name2:  ## 提取文件夹名的头部
                    isout = False  ## 保证跳出循环
                    break
            if isout or num > 100:  ## tensor 文件夹不带时间戳时可用
                break

        if num > 99: raise IndexError(
            'log of tensorboard auto change path is out of max[999]! Now path is: %s' % (path + '%02d' % num))
        # name2 = name2 + ' {}'.format(tm_str)  # 文件夹名加入时间戳
        tensor_path = os.path.join(father_path, name2)
        os.makedirs(tensor_path, exist_ok=True)
        return tensor_path

    def prepare_tensor_path(self, tensor_path):
        """  创建文件名，以提示对应的csv文件 """
        tm_str = os.path.basename(self.config_file)[:-5]
        tm_str = tm_str[len('config '):]  ## 提取时间戳
        file = os.path.join(tensor_path, 'train data {}'.format(tm_str))
        f = open(file=file, mode='w')
        f.close()
        print( 'tensor_path:', tensor_path)
        # exit()


    def add_scalars_(self, main_tag: str, scalars: {}, step: int, tolerant=False):
        if main_tag not in self.headers.keys():
            raise "main_tag: [%s] should be one of : %s" % (main_tag, str(self.headers.keys()))
        if not tolerant and scalars.keys() != self.headers[main_tag]:
            raise "scalars: [{}] should be one of : {}".format(scalars.keys(), self.headers[main_tag])

        if self.logWriter is None and self.save_log:
            # path = os.path.join(os.path.dirname(self.csv_log_file), 'tensor')
            # path = self.update_auto_pathname(path)
            path = self.tensor_log_file
            # self.prepare_tensor_path(path)
            self.logWriter = SummaryWriter(path, comment='log', )
            print(bcolors.tip + '[log]:\tTensorboardX log path is:' + bcolors.ENDC, path, flush=True)
            print(f'{bcolors.tip}\t you can open a terminal and use the command ['
                  f'{bcolors.error}tensorboard --logdir={path}{bcolors.tip}] for visualization' + bcolors.ENDC,
                  flush=True)
            if self.write_csv:
                print(bcolors.tip + '[log]:\tThe training process data are stored as csv in:' + bcolors.ENDC,
                      self.csv_log_file, flush=True)

        if self.save_log:
            if self.tensorboard_mode.lower() == 'train-test':
                """ 根据 headers 的模式分组显示 """
                self.logWriter.add_scalars(main_tag=main_tag, tag_scalar_dict=scalars, global_step=step)

            elif self.tensorboard_mode.lower() == 'loss-acc':
                self.logWriter_add_scalar(main_tag, scalars, step, tolerant)

        ''' 记录全流程数据 '''
        isold = False
        try:
            idx = self.data_buff['idx'].index(step)  ## 检索日志的索引号，可用于间隔保存模式
            isold = True
        except:
            self.data_buff['idx'].append(step)
            idx = len(self.data_buff['idx']) - 1

        for name in self.data_buff.keys():  ## 保证所有数据长度相同-->才能写入csv/excel
            if name == 'idx': continue
            main_tag0, sub_head = str(name).split(self.separator)
            if main_tag0 != main_tag:  ## 传入数据不是当前项
                try:
                    dat = self.data_buff[name][idx]  ## 判断原来是否有数据，【使用原数据】
                except:
                    self.data_buff[name].append(np.nan)  ## 原来没有数据，添加：空字符
                continue
            try:
                dat = scalars[sub_head]  ## 提取传入数据
            except:
                dat = np.nan  ## 传入数据没有该项，默认 nan
            try:
                self.data_buff[name][idx] = dat  ## 覆盖原有数据
            except:
                self.data_buff[name].append(dat)  ## 添加新数据
        ''' 打印输出 '''
        # if iscomplete:
        self.format_print_train_state(-1, isold)
        ''' 定期写入 csv 文件 '''
        if self.write_csv and len(self.data_buff[self.csv_head[0]]) % 20 == 0:
            self.write_csv_all(None, False)
        pass

    def getMax(self, main_tag: str, scalar: str, rm_last=False):
        name = main_tag + self.separator + scalar
        if rm_last:
            if len(self.data_buff[name]) <= 1: return 0
            dat = np.array(self.data_buff[name][:-1]).max()
        else:
            if len(self.data_buff[name]) <= 0: return 0
            dat = np.array(self.data_buff[name]).max()
        # print('getMax:', dat)
        return dat

    def getMin(self, main_tag: str, scalar: str, rm_last=False):
        name = main_tag + self.separator + scalar
        if rm_last:
            if len(self.data_buff[name]) <= 1: return 0
            dat = np.array(self.data_buff[name][:-1]).min()
        else:
            if len(self.data_buff[name]) <= 0: return 0
            dat = np.array(self.data_buff[name]).min()
        # print('getMin:', dat)
        return dat

    def get_data(self, main_tag: str, scalars: list, rm_last=False):
        try:
            idx_list = []
            for sca in scalars:
                idx_list.append(main_tag + self.separator + sca)
            dat = {}
            for i, sca in enumerate(scalars):
                if rm_last:
                    dat[sca] = self.data_buff[idx_list[i]][:-1]
                else:
                    dat[sca] = self.data_buff[idx_list[i]]
            return dat
        except:
            raise KeyError('main_tag and scalars should be the same as {}'.format(self.headers))

    def write_csv_all(self, file=None, ishow=True):
        if file is None: file = self.csv_log_file
        df = pd.DataFrame(data=self.data_buff)
        # df.to_csv(file, index=False)
        df.to_excel(file, index=None)
        if ishow: print('[log]:\tcsv file is saved in:', file, flush=True)
        pass

    def format_print_train_state(self, idx=-1, isold=False):
        ''' 打印输出
            idx: 输出索引！ 默认【-2】是输出上一拍，确保整体信息已更新完整
                 【-1】：实时输出，但信息可能不完整
        '''
        try:
            fprint = ''
            head = '' if isold is True else '\n'  ## 不是old的情况下，头端需要换行
            end = ''  # if isold is False else '\n'   ## 不是old的情况下，末端先不换行，避免后续是old 数据
            for name in self.headers.keys():
                fprint += '\n\t' + name + ': '
                for sub_head in self.headers[name]:
                    k = name + self.separator + sub_head
                    dat = self.data_buff[k][idx]
                    fprint += '{}:{:.08f} | '.format(sub_head, dat)
            epoch = self.data_buff['idx'][idx]
            if len(fprint) < 140: fprint = fprint.replace('\n\t', '\t ')  # 太短就单行显示
            # print('format_print:', len(fprint), iscomplete, isold, '---{}---'.format(end) )
            print('{}\rEpoch:{}'.format(head, epoch + 1) + fprint, end=end, flush=True)
        except:
            pass

    def save_config(self, config: {}):
        if not self.save_log: return
        self.make_tensor_path()
        """ 保存程序基本配置 """
        df = YamlHandler(self.config_file).write_yaml(config)
        print(bcolors.tip + '[log]:\tProject config file is saved in:' + bcolors.ENDC, self.config_file, flush=True)
        pass

    def logWriter_add_scalar(self, main_tag: str, scalars: {}, step: int, tolerant=False):
        """ 变更tensorboard的网页分组显示方式 【loss-acc】：不同数据集的同loss显示在一个图上 """
        # headers = {'train':['loss'], 'test':['loss'], 'log':['lr', 'time']}
        if main_tag.lower() in ['train', 'test', 'val']:
            for new_main_tag in scalars.keys():
                new_scalars = main_tag + ' ' + new_main_tag
                self.logWriter.add_scalars(main_tag=new_main_tag, tag_scalar_dict={new_scalars: scalars[new_main_tag]},
                                           global_step=step)
        else:
            for new_main_tag in scalars.keys():
                new_scalars = main_tag + '/' + new_main_tag
                self.logWriter.add_scalar(tag=new_scalars, scalar_value=scalars[new_main_tag], global_step=step)
        pass

    def __del__(self):
        if not self.write_csv: return
        try:
            issave = False
            for name in self.data_buff.keys():
                if len(self.data_buff[name]) > 0:
                    issave = True
            if issave: self.write_csv_all(None, False)
        except:
            pass
        try:
            self.logWriter.close()
        except:
            pass


## 获取类内的所有变量和值
def get_variation_of_Class(clas, variate=True, func=False):
    """ Get all variation of the clas,
        return a dict:{variation: value}
    """
    members = [attr for attr in dir(clas) if not callable(getattr(clas, attr)) and not attr.startswith("__")]  #
    res = {}
    for key in members:
        value = getattr(clas, key)
        res[key] = value
    return res


def delete_path(path):
    """ delete all files of the path"""
    try:
        if os.path.isdir(path):
            os.rmdir(path)
        else:
            os.remove(path)
    except:
        files = os.listdir(path)
        for f in files:
            delete_path(os.path.join(path, f))
        os.rmdir(path)  ## 报错之后，需要再度删除自己
        # print('remove folder:', path)


## 删除某一次训练的日志记录 对应：Summary_Log
def delete_all_log_record_file(root, idx: list = None, tm=None, rm_config=True, rm_model=True, rm_tensor=True,
                               rm_csv=True):
    def delete_all_log_record_file(root, idx: int = None, tm=None, rm_config=True, rm_model=True, rm_tensor=True,
                                   rm_csv=True):
        """ 日志删除，不可恢复
            可通过次数索引 idx（从tensorboard中获取具体时间信息），或者 tm 定位日志
            单独控制 config、model、tensorboard、csv 的删除
        """

        """ 日志定位 """
        if idx == 0: return -1
        if tm is None and idx is None: return -1
        tar_tm = tm  # 删除的指定时间戳  eg. 2022-10-11 20.12.11
        rm_files = []
        if tar_tm is None and idx is not None:
            name = 'tensor' + '%02d' % idx if idx > 1 else 'tensor'

            ''' 如果tensor文件夹名称包含时间戳 '''
            # p1 = os.path.join(root, 'log')
            # try:
            #     tensorfiles_list = os.listdir(p1)
            # except: return
            # for f in tensorfiles_list:
            #     p2 = os.path.join(p1, f)
            #     if not os.path.isdir(p2): continue
            #     name_f = str(f).split(' ')[0]  # 提取文件夹名称的第一部分 tensor与时间戳的间隔必须是空格
            #     # print('====[{}]    [{}]'.format(name, name_f))
            #     if name == name_f:
            #         name = f  ## 提取到整个文件夹名称
            #         """ 从文件夹名中提取时间戳 """
            #         tar_tm = f[len(name_f)+1:]
            #         break
            # if tar_tm is None: return
            # if len(tar_tm)<1 : ## 旧版本保护，不能跳出去
            """ 从文件夹的文本文件名中提取时间戳 """  ## 旧版本保护
            path1 = os.path.join(root, 'log', name)  # 确定 tensor文件夹名称
            try:
                files = os.listdir(path1)  ## 获取 tensor文件夹中的 所有文件， 目的是找到 'train data'开头的时间记录文件
            except:
                return 0
            for f in files:
                if f.startswith('train data'):  # 获得时间戳
                    tar_tm = f[len('train data '):]
                    break
        tar_tm = str(tar_tm)  # 获得时间戳
        if rm_csv or rm_tensor:
            path1 = os.path.join(root, 'log')
            files = os.listdir(path1)
            for f in files:
                new_f = os.path.join(path1, f)
                if os.path.isfile(new_f) and tar_tm in f and rm_csv:  ## 删除 csv
                    rm_files.append(new_f)
                    continue
                elif os.path.isdir(new_f) and rm_tensor:  ## 删除tensorboard
                    fs = os.listdir(new_f)
                    fs.reverse()
                    is_in = False
                    for ff in fs:
                        if tar_tm in ff:
                            is_in = True
                            break
                    if is_in:  rm_files.append(new_f)
                    # print(fs)
                    # if str(tar_tm) in '{}'.format(fs):        ## 删除tensorboard
                    #     rm_files.append(new_f)
                    # else:
                    #     print('=====', '{}'.format(fs))

        if rm_model:
            path1 = os.path.join(root, 'model')
            files = os.listdir(path1)
            for f in files:
                if tar_tm in f and rm_model:  # 删除 model
                    rm_files.append(os.path.join(path1, f))

        if rm_config:
            files = os.listdir(root)
            for f in files:
                if f.endswith('yaml') and tar_tm in f:  # 删除 config
                    rm_files.append(os.path.join(root, f))
        print('\033[031m即将从  {}  删除以下文件[{}]:'.format(root, idx))
        for i in rm_files:
            print('\t',i)
        x = input('\t是否继续[y/n] or [1/0]:\033[0m'.format(), )
        if str(x).lower() != 'y' and str(x) != '1': return

        # from shutil import rmtree
        # print('delete_all_log_record_file:', rm_files, tar_tm)
        for i in rm_files:
            delete_path(i)
            # rmtree(i, ignore_errors=False, onerror=None)  ## 彻底删除
            print(f'[log]: remove file [{i}]')

    if len(idx) == 0: return
    # x = input('\033[031m即将从 {}\n\t删除以下文件 {} 是否继续[y/n] or [1/0]:\033[0m'.format(root, idx), )
    # if str(x).lower() != 'y' and str(x) != '1': return
    for i in idx:
        delete_all_log_record_file(root, i, tm, rm_config, rm_model, rm_tensor, rm_csv)


import matplotlib.pyplot as plt
from matplotlib import rcParams


class Visual_Model_Predict:
    def __init__(self, ):
        self.data_buff = {}
        self.separator = ' '
        pass

    def add_data_classify(self, main_tag: str = None, data: {} = None):
        self.add_data_series(main_tag, data)
        pass

    def add_data_series(self, main_tag: str = None, data: {} = None):
        for name in data.keys():
            if main_tag is not None:
                name2 = main_tag + self.separator + name
            else:
                name2 = name

            dat = data[name]
            try:
                if len(self.data_buff[name2]) == 0:
                    self.data_buff[name2] = list(dat)
                    # print('==1==', dat.shape, len(self.data_buff[name2]))
                else:
                    self.data_buff[name2] += list(dat)
                    # print('==2==', dat.shape, len(self.data_buff[name2]))
            except:
                self.data_buff[name2] = list(dat)  ## buffer中不存在该列，dict会自动创建
                # print('==0==', dat.shape, len(self.data_buff[name2]))
        pass

    # def draw_figure(self, main_tag:str=None, keys:list=None):
    #     # t = threading.Thread(target=self.v, args=(main_tag, keys))
    #     # t.start()   ##  可能无法显示
    #     self.draw_figure_line(main_tag, keys)

    def draw_figure_line(self, main_tag: str = None, keys: list = None):
        # idx_list = []
        # for name in keys:
        #     if main_tag is not None: name2 = main_tag + self.separator + name
        #     else: name2 = name
        #     idx_list.append(name2)
        #
        # data_list = []
        # sorted_idx = None
        # for idx in idx_list:
        #     dat = self.data_buff[idx]
        #     data_list.append(dat)
        #     if sorted_idx is None: sorted_idx = np.array(dat).argsort(axis=0).reshape(-1)
        #     # print(np.array(dat).argsort(axis=0))
        # # sorted_idx = np.array(sorted_idx, dtype=np.int)
        # # print(sorted_idx, data_list[0])
        data_list, sorted_idx, idx_list = self.__get_data(main_tag, keys)

        plt.figure('draw_figure_line')
        x = np.arange(0, len(sorted_idx))
        for i, d in enumerate(data_list):
            d = np.array(d).reshape(-1, 1)
            # print(d.shape, sorted_idx.shape, d[sorted_idx].shape)
            plt.plot(x, d[sorted_idx], label=str(idx_list[i]))
            plt.draw()
            plt.pause(0.1)
        plt.legend()
        plt.grid()
        plt.draw()
        plt.pause(0.1)

        pass

    def __get_data(self, main_tag: str = None, keys: list = None):
        idx_list = []
        for name in keys:
            if main_tag is not None:
                name2 = main_tag + self.separator + name
            else:
                name2 = name
            idx_list.append(name2)

        data_list = []
        sorted_idx = None
        for idx in idx_list:
            dat = self.data_buff[idx]
            data_list.append(dat)
            if sorted_idx is None: sorted_idx = np.array(dat).argsort(axis=0).reshape(-1)
            # print(np.array(dat).argsort(axis=0))
        return data_list, sorted_idx, idx_list

    def __classify_to_matrix(self, main_tag: str = None, keys: list = None):
        if len(keys) != 2: raise ValueError('len of keys must be: len(keys)=2! Now keys is', keys)
        data_list, sorted_idx, idx_list = self.__get_data(main_tag, keys)
        # print(len(data_list[0]))
        # print(data_list[0])
        label = np.array(data_list[0], dtype=np.int)
        predict = np.array(data_list[1], dtype=np.int)
        # print('***',  len(label[label==0]), label[label==1])
        # print('===', np.sum(label==0), np.sum(label==1), np.sum(label==2), np.sum(label==3))
        length = label[sorted_idx[-1]] + 1
        length2 = max(predict.reshape(-1)) + 1
        length = max(length, length2)
        matrix = np.zeros([length, length], dtype=np.int)
        for i, t_label in enumerate(label):
            pred = predict[i]
            matrix[t_label, pred] += 1  # [t_label, pred]  纵坐标为真标签，横坐标为预测结果
        # print(matrix.sum(axis=1), matrix.sum(axis=0) )
        return matrix

    def draw_figure_matrix(self, main_tag: str = None, keys: list = None):

        matrix = self.__classify_to_matrix(main_tag, keys)

        # matrix = np.array(
        #     [[451, 1, 12, 6, 1, 3, 5, 2],
        #      [18, 451, 25, 19, 24, 14, 7, 2],
        #      [41, 27, 487, 2, 15, 2, 24, 3],
        #      [14, 20, 4, 395, 7, 16, 15, 5],
        #      [1, 8, 30, 25, 421, 16, 14, 14],
        #      [13, 18, 1, 15, 13, 455, 18, 19],
        #      [19, 7, 12, 17, 4, 21, 352, 15],
        #      [15, 23, 31, 15, 3, 9, 15, 458]
        #      ], dtype=np.int)  # 输入特征矩阵
        matrix = np.array(matrix, dtype=np.int)
        matrix_norm = matrix.sum(axis=1).reshape(-1, 1)
        matrix_norm = matrix / matrix_norm
        matrix_norm_str = np.array(matrix_norm.copy(), dtype=np.str)

        acc_total = 0
        rows, clos = matrix_norm.shape
        for i in range(rows):
            for j in range(clos):
                matrix_norm_str[i, j] = ("%.02f" % (matrix_norm[i, j] * 100))
                if i == j: acc_total += matrix[i, j]
        # print(matrix_norm_str)
        acc_total = acc_total / np.sum(matrix.reshape(-1))
        print('[draw_figure_matrix] acc_total:', acc_total, )
        """ 绘制混淆矩阵图 """
        plt.figure('draw_figure_matrix')
        plt.clf()
        config = {
            "font.family": 'Times New Roman',  # 设置字体类型
        }
        rcParams.update(config)
        plt.imshow(matrix_norm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)  # 按照像素显示出矩阵  nearest
        # (改变颜色：'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds','YlOrBr', 'YlOrRd',
        # 'OrRd', 'PuRd', 'RdPu', 'BuPu','GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn')
        # plt.title('confusion_matrix')
        plt.colorbar()
        y_tick_marks = np.arange(matrix_norm.shape[0])
        x_tick_marks = np.arange(matrix_norm.shape[1])
        plt.xticks(ticks=x_tick_marks, labels=x_tick_marks, fontsize=12)
        plt.yticks(ticks=y_tick_marks, labels=y_tick_marks, fontsize=12)

        """ 矩阵图对应位置显示数字 【必须在设置了坐标轴索引之后才能在 text中定位位置 】  """
        rows, clos = matrix_norm.shape
        for y in range(rows):
            for x in range(clos):
                # if (y == x):
                #     plt.text(x, y - 0.18, format(matrix[y, x]), va='center', ha='center', fontsize=10,
                #              # color='white',
                #              weight=5)  # 显示对应的数字
                #     plt.text(x, y + 0.18, matrix_norm_str[y, x], va='center', ha='center', fontsize=10,
                #              # color='white',
                #              weight=5)
                #     # print(f'\t[{x},{y}] --- {format(matrix[y, x])}  {matrix_norm_str[y, x]}')
                # else:
                #     plt.text(x, y - 0.18, format(matrix[y, x]), va='center', ha='center', fontsize=10)  # 显示对应的数字
                #     plt.text(x, y + 0.18, matrix_norm_str[y, x], va='center', ha='center', fontsize=10)
                #     # print(f'[{x},{y}] --- {format(matrix[y, x])}  {matrix_norm_str[y, x]}')
                if matrix_norm[y, x] > 0.75:
                    color = 'white'
                    weight = 5
                else:
                    color = 'black'
                    weight = 2
                plt.text(x, y - 0.20, format(matrix[y, x]), va='center', ha='center', fontsize=10,
                         color=color, weight=weight)  # 显示对应的数字
                plt.text(x, y + 0.20, matrix_norm_str[y, x], va='center', ha='center', fontsize=10,
                         color=color, weight=weight)

        plt.ylabel('True label', fontsize=14)
        plt.xlabel('Predict label', fontsize=14)
        plt.tight_layout()
        # plt.draw()
        # plt.pause(0.1)
        # plt.savefig('混淆矩阵.png')
        pass

    def __del__(self):
        # try: plt.pause(60*5)
        # except: pass
        pass
