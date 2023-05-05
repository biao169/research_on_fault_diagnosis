import datetime
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn, optim
from utils_tool.log_utils import Summary_Log

def func_param_to_dict(**kwargs):
    return kwargs

class Train_base:
    def __init__(self):
        self.epochNum = 0
        self.resultPath = os.getcwd()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # ("cpu") #
        self.modelPath = os.path.join(self.resultPath, 'model')
        self.startTime = datetime.datetime.now().strftime('%Y-%m-%d %H.%M.%S')
        self.train_dataLoader = None  # classmethod
        self.test_dataLoader = None  # classmethod
        self.summary_log = None  ## Summary_Log(path=self.resultPath, headers=None, write_csv=True,tm_str=self.startTime, save_log=True)

        self.net = nn.Module()
        self.opti_log = {}
        self.config:{} = None
        pass

    def config_make(self):
        args = {
            'device': str(self.device),
            'resultPath': self.resultPath,
            'epochNum': self.epochNum,
        }
        if self.train_dataLoader is not None:
            try:
                path = self.train_dataLoader.dataset.config_make()
            except:
                try: path = self.train_dataLoader.dataset.root
                except: path = self.train_dataLoader.dataset.path
            arg2 = {'train_loader': {
                'miniBatch': self.train_dataLoader.batch_size,
                'dataset': path
            }}
            args = {**args, **arg2}

        if self.test_dataLoader is not None:
            try:
                path = self.test_dataLoader.dataset.config_make()
            except:
                try: path = self.test_dataLoader.dataset.root
                except: path = self.test_dataLoader.dataset.path
            arg2 = {'test_loader': {
                'miniBatch': self.test_dataLoader.batch_size,
                'dataset': path
            }}
            args = {**args, **arg2}

        try:
            args = {**args, **self.opti_log}
        except: pass
        return args
        pass

    def training_init_save(self):
        self.opti_log['network'] = {'name': str(self.net.__class__),
                                    'file path': self.net.___file__,
                                    'log': self.net.log}

        args = self.config_make()
        for key in self.config:
            if key in args.keys(): continue
            args[key] = self.config[key]
        self.summary_log.save_config(args)
        os.makedirs(self.modelPath, exist_ok=True)
        print(f'[base]: train batchSize={self.train_dataLoader.batch_size}, \ttrainLoader num={len(self.train_dataLoader)}')

    def optimizer(self, net=None,  opt_name:str='Adam', kwargs:{}=None):
        # lr = 1e-4, momentum = 0.9, weight_decay = 5 * 1e-4
        # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False
        if net is None: net = self.net
        if opt_name.lower() == 'sgd':
            optimizer = optim.SGD(net.parameters(), **kwargs)
        else:
            optimizer = optim.Adam(net.parameters(), **kwargs)
        self.opti_log['optimizer'] = {opt_name: kwargs}
        return optimizer

    def optimizer_multi_nets_parameters(self, params=None, opt_name: str = 'Adam', kwargs: {} = None):
        # lr = 1e-4, momentum = 0.9, weight_decay = 5 * 1e-4
        # lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False  params={'params': param_groups}
        """
        :param params: 可以是多个网络参数，用list['params'：parameters]组合
        :param opt_name: SGD // Adam
        :param kwargs:  {lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0,]
        :return: optimizer
        """
        """ params = [{'params': model.parameters()}, {'params': lossnet.parameters(), 'lr': 1e-4}]  """
        if params is None: params = self.net.parameters()
        if opt_name.lower() == 'sgd':
            optimizer = optim.SGD(params, **kwargs)
        else:
            optimizer = optim.Adam(params, **kwargs)
        self.opti_log['optimizer'] = {opt_name: kwargs}
        return optimizer

    def scheduler(self, optimizer, sched_name: str = 'ReduceLROnPlateau', kwargs:{} = None):
        if sched_name.lower()=='ReduceLROnPlateau'.lower():
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **kwargs)
            # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=30,
            #                                                        eps=1e-10, cooldown=5,
            #                                                        verbose=False)  ## val_acc  , min_lr=1e-10
            # kwargs = func_param_to_dict(mode='min', factor = 0.5, patience = 30, eps = 1e-10, cooldown = 5, verbose = False)

        else:
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **kwargs)

        self.opti_log['scheduler'] = {**kwargs}
        return scheduler

    def get_learn_rate(self, optimizer):
        res = []
        for group in optimizer.param_groups:
            res.append(group['lr'])
        return res

    def set_learn_rate(self, optimizer, lr=0.01):
        for i, param_group in enumerate(optimizer.param_groups):
            param_group['lr'] = lr
        return optimizer

    def saveModel_onlyOne(self, net=None, epoch=0, name=''):
        if not self.summary_log.save_log: return
        modelPath = self.modelPath
        name0= name
        if net is None: net= self.net
        if name =='':
            new_name = 'checkpoint_{}_epo[{}].model'.format(self.startTime, epoch)
            rm_name = '].model'   ##  避免误删
        else:
            new_name = 'checkpoint_{}_epo[{}]_{}.model'.format(self.startTime, epoch, name)
            rm_name = '_{}.model'.format(name)  ##  避免误删

        filelist = os.listdir(modelPath)
        for filename in filelist:
            if filename.endswith('.model'):
                if filename.startswith('checkpoint') and rm_name in filename and self.startTime in filename:
                    file = os.path.join(modelPath, filename)  # 最终参数模型
                    os.remove(file)
                    # print('saveModel_onlyOne:','remove', file)
            # if filename.startswith(('scriptmodule_{}.pt').format(note)):
            #     file = os.path.join(modelPath, ('scriptmodule_{}.pt').format(note))
            #     os.remove(file)
        filepath = os.path.join(modelPath, new_name)  # 最终参数模型
        state = {'Net': net.state_dict(), 'epoch': epoch}
        torch.save(state, filepath)
        print(f'\t[base]: --- torch.save [{name0}] model:', filepath)

    def load_model_weight(self, net, name='', **kwargs):  # param
        files = os.listdir(self.modelPath)
        files.reverse()
        for f in files:
            if str(f).endswith('.model') and name in str(f):
                model_file = os.path.join(self.modelPath, f)
                print('\t[base]: loading model weight:', model_file)
                model_dict = torch.load(model_file)  # param.
                net.load_state_dict(model_dict["Net"])
                epoch = model_dict['epoch']
                return net, epoch
        print('\t[base]: load model weight: [fail]', self.modelPath, files)
        return net, 0

    def load_model_weight_file(self, net, file='', **kwargs):  # param
        model_file = os.path.join(self.modelPath, file)
        print('\t[base]: loading model weight:', model_file)
        model_dict = torch.load(model_file)  # param.
        net.load_state_dict(model_dict["Net"])
        epoch = model_dict['epoch']
        return net, epoch

    def load_model_weight_auto(self, net, **kwargs):
        if 'file' in kwargs:
            net, epoch = self.load_model_weight_file(net, file=kwargs['file'])
        else:
            net, epoch = self.load_model_weight(net, **kwargs)
        return net, epoch

    '''
    def control_online(self, stop_epoch, optimizer):
        res = controlManage()
        stop_train = False
        save_model = False
        test_model = False
        if res['epoch'] is not None:
            print('原始 epoch={} 更替为：{}'.format(stop_epoch, res['epoch']))
            stop_epoch = res['epoch']

        if res['lr'] is not None:
            new_lr = res['lr']
            optimizer = self.set_learn_rate(optimizer, new_lr)
        if res['save_model']:
            save_model = True
        if res['test_model']:
            test_model = True

        if res['exit'] is True:
            print('\n----------------------------------------------------\n',
                  '---------------  收到退出指令-->退出  ！！！！ ----------------------')
            stop_train = True
            if res['remove_txt'] is True:
                self.delete_log_model(ismodel=False)
                # shutil.move(txtPath + 'train{}.txt'.format(note), outremovePath)
                # shutil.move(txtPath + 'val{}.txt'.format(note), outremovePath)
                # shutil.move(txtPath + 'test{}.txt'.format(note), outremovePath)
                # os.remove(txtPath + 'train{}.txt'.format(note))
                # os.remove(txtPath + 'val{}.txt'.format(note))
                # os.remove(txtPath + 'test{}.txt'.format(note))
                if res['remove_model'] is True:
                    self.delete_log_model(ismodel=True)
                    self.exit = True
            # print('-----------  清除数据文件完成  移动到文件夹:{} -----------------\n\n'.format(outremovePath))
        # stop_epoch, optimizer, stop_train, save_model, test_model
        res = {'stop_epoch': stop_epoch, 'optimizer': optimizer, 'stop_train': stop_train,
               'save_model': save_model, 'test_model': test_model}
        return res

    def delete_log_model(self, logName=None, ismodel=True, islog=True):
        if logName is None: logName = self.log_name
        print('------  清理文件：  -----------------')  # ,self.netPath, self.log_name
        if ismodel:
            try:
                file1 = logName.format('checkpoint')
                model_list = os.listdir(self.netPath)
                for model in model_list:
                    if model.endswith('.model') and model.startswith(file1):
                        os.remove(os.path.join(self.netPath, model))
                        print('\t', model)
            except:
                pass
            try:
                file2 = logName.format('log') + '.py'
                os.remove(os.path.join(self.netPath, file2))
                print('\t', file2)
            except:
                pass
        if islog:
            try:
                file3 = logName.format('data') + '.csv'
                os.remove(os.path.join(self.netPath, file3))
                print('\t', file3)
            except:
                pass
        pass

    # def record_training_log(self, **kwargs):
    #     file1 = self.log_name.format('log') + '.py'
    #     file1 = os.path.join(self.netPath, file1)
    #     log = record2Text()
    #     if log != '':
    #         with open(file=file1, mode='a', encoding='utf-8') as f:
    #             f.write(log)
    #     self.record_dataloader(file1)
    #     self.record_netconfig(**kwargs)
    #     log = self.record_add_log()
    #     if log != '':
    #         with open(file=file1, mode='a', encoding='utf-8') as f:
    #             f.write(log)

    def record_netconfig(self, shuffle=False, drop_last=False, tips=None, dataloader_shuffers='', dataset_shuffers=''):
        file1 = self.log_name.format('log') + '.py'
        file1 = os.path.join(self.netPath, file1)
        with open(file=file1, mode='a', encoding='utf-8') as f:
            f.write("'''" + self.netname + "'''\n")
            f.write('trainbatch_size = ' + str(self.trainbatch_size) + '\n')
            f.write('testbatch_size = ' + str(self.testbatch_size) + '\n')
            f.write('shuffle = ' + str(shuffle) + '\n')
            f.write('drop_last = ' + str(drop_last) + '\n')
            f.write('tips = ' + str(tips) + '\n')
            f.write('\ndataloader_shuffers = ' + str(dataloader_shuffers) + '\n')
            f.write('dataset_shuffers = ' + str(dataset_shuffers) + '\n')
            f.write("\n\n## -- train_base_class: [{}] --\n\n".format(self._class_file_))
            with open(self._class_file_, mode='r', encoding='utf-8') as f2:
                # print('__file__:', self._class_file_)
                start_p = False
                lines = f2.readlines()
                for line in lines:
                    if 'def optimizer(self)' in line: start_p = True
                    if 'return optimizer' in line:
                        f.write(line)
                        break
                    if start_p:
                        f.write(line)

        pass

    def record_dataloader(self, file1):
        with open(file=file1, mode='a', encoding='utf-8') as f:
            f.write("\n\ntrainDataLoader.dataSet = " + str(self.train_loader.dataset.__class__.__name__) + "\n\n")

    def record_add_log(self):  ##->str
        return ''
    '''
