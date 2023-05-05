import random
import torch
import torch.utils
from torch.utils import data
import numpy as np
import os

from dataset.dataset_CRWU import DataSet_CRWU_MultiFile
from networks.classic_network.dcnn import dcnn as Net
import networks.classic_network.dcnn.train_dcnn as Train
from utils_tool.log_utils import YamlHandler, delete_all_log_record_file

from report_make.excel_report import Make_Report

"""
    本部分程序只用于训练 复现 DRSN 

"""

config = YamlHandler('config.yaml').read_yaml()


# print(config)
# exit()
def seed_torch(seed=20):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# torch.use_deterministic_algorithms(True)  # 有检查操作，看下文区别
seed_torch(config['seed'])

from utils_tool.signal_normalization import Wavelet_Fourier_transform_numpy
from torchvision import transforms
import cv2 as cv

class DCNN_Dataset(data.Dataset):  ##
    """ 使用 data.random_split 划分数据集后的 dataset 不具备类的功能
        在train_ways/base.py中无法调用 args 的属性，需要重构
        *** data.random_split 划分数据集，各类样本可能不均衡 ***
    """

    def __init__(self, dataset, args):
        super(DCNN_Dataset, self).__init__()
        self.dataset = dataset
        self.args = args
        self.transforms = transforms.Compose([
            # transforms.Resize((128, 128)),  ##  需要  PIL Image
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456], std=[0.229, 0.224])
        ])

    def __getitem__(self, item):
        DE, FE, BA, label = self.dataset[item]
        """ DE = torch.tensor(DE, dtype=torch.float).reshape(1, -1) """
        DE = Wavelet_Fourier_transform_numpy(DE.reshape(-1).numpy())
        FE = Wavelet_Fourier_transform_numpy(FE.reshape(-1).numpy())
        DE = cv.resize(DE, (128,128))
        FE = cv.resize(FE, (128, 128))
        data = np.stack([DE, FE], axis=2)  ## [C W H]
        data = self.transforms(data)
        return data, label

    def __len__(self):
        return len(self.dataset)

    def config_make(self):
        return self.args
        pass


def dcnn_config(config=config):
    root = r'I:\python_datasets\CRWU\CRWU_original'
    config['train_loader']['dataset']['path'] = root
    config['test_loader']['dataset']['path'] = root
    load_files = config['crwu_set1']  ## crwu_set1_2  crwu_set1
    config['optimizer'] = {}
    config['scheduler'] = None
    config['optimizer']['Adam'] = dict(lr=0.0001, betas=[0.9, 0.999], weight_decay=0.0001)  # momentum=0.9, weight_decay=0.0001
    config['batchSize'] = 32
    config['train_loader']['dataset']['miniBatch'] = 32
    config['test_loader']['dataset']['miniBatch'] = 32
    # config['scheduler']['StepLR'] = dict(step_size=40, gamma=0.1, last_epoch=-1)
    return config, load_files


def dcnn_config2(config=config):
    """ Only for evaluation in another condition or fault """
    load_files = config['crwu_set2']
    config, _ = dcnn_config()

    return config, load_files


def official_train():
    # Fixme
    net = Net.DCNN_MutilChannel(in_channel=2, out_channel=10)
    config, load_files = dcnn_config()
    config['resultPath'] = os.path.join(config['resultPath'], r'{}_official'.format(net.log))

    train_dataset, test_dataset = DataSet_CRWU_MultiFile().get_train_and_test_set_ordered(
                        config['train_loader']['dataset']['path'], load_files,
                        train_test_rate=0.5, repeat_win=0.85, window=1024)
    # train_dataset, test_dataset = DataSet_CRWU_MultiFile().get_train_and_test_set_ordered_enhance(
    #                     root=config['train_loader']['dataset']['path'],
    #                     load_files=load_files,
    #                     train_test_rate=0.5, repeat_win=0.85, window=1024,
    #                     mask_win=0,
    #                     mask_start_idx=0,
    #                     snr=None)  # cate='test',  #  int(1024 * 0.6) random DataSet_CRWU_Signal_Process
    """ 函数变更，采用连续小波傅里叶变换 """
    train_dataset = DCNN_Dataset(train_dataset, train_dataset.args)
    test_dataset = DCNN_Dataset(test_dataset, test_dataset.args)
    print(f'train_dataset len = {len(train_dataset)},\t test_dataset len = {len(test_dataset)}')

    trainloader = data.DataLoader(dataset=train_dataset, batch_size=config['train_loader']['dataset']['miniBatch'],
                                  shuffle=config['train_loader']['shuffle'], drop_last=False, )
    testloader = data.DataLoader(dataset=test_dataset, batch_size=config['test_loader']['dataset']['miniBatch'],
                                 shuffle=config['test_loader']['shuffle'], drop_last=False, )

    """ 删除日志文件 """
    delete_all_log_record_file(root=config['resultPath'], idx=[], tm=None, rm_config=True, rm_csv=True,
                               rm_model=True, rm_tensor=True)
    config['epochNum'] = 500
    train = Train.Train(net=net, train_dataLoader=trainloader, test_dataLoader=testloader,
                        config=config, write_csv=True, save_log=True, tensorboard_mode='loss-acc',
                        new_thread=True)
    train.training_mode(pretrain=False, datasetName='crwu')
    pass


def designed_train():
    # Fixme
    net = Net.DCNN_MutilChannel(in_channel=2, out_channel=10)
    config, load_files = dcnn_config()
    config['resultPath'] = os.path.join(config['resultPath'], r'{}_designed'.format(net.log))

    train_dataset, test_dataset = DataSet_CRWU_MultiFile().get_train_and_test_set_ordered_enhance(
                                    root=config['train_loader']['dataset']['path'],
                                    load_files=load_files,
                                    train_test_rate=0.5, repeat_win=0.85, window=1024,
                                    mask_win=int(1024 * 0.8),
                                    mask_start_idx=None,
                                    snr=None)
    config['train way'] = dict(mask_win=0.8, mask_start_idx=None, snr=None)
    """ 函数变更，采用连续小波傅里叶变换 """
    train_dataset = DCNN_Dataset(train_dataset, train_dataset.args)
    test_dataset = DCNN_Dataset(test_dataset, test_dataset.args)
    print(f'train_dataset len = {len(train_dataset)},\t test_dataset len = {len(test_dataset)}')

    trainloader = data.DataLoader(dataset=train_dataset, batch_size=config['train_loader']['dataset']['miniBatch'],
                                  shuffle=config['train_loader']['shuffle'], drop_last=False, )
    testloader = data.DataLoader(dataset=test_dataset, batch_size=config['test_loader']['dataset']['miniBatch'],
                                 shuffle=config['test_loader']['shuffle'], drop_last=False, )

    """ 删除日志文件 """
    delete_all_log_record_file(root=config['resultPath'], idx=[], tm=None, rm_config=True, rm_csv=True,
                               rm_model=True, rm_tensor=True)
    config['epochNum'] = 500
    train = Train.Train(net=net, train_dataLoader=trainloader, test_dataLoader=testloader,
                        config=config, write_csv=True, save_log=True, tensorboard_mode='loss-acc',
                        new_thread=True)
    train.training_mode(pretrain=False, datasetName='crwu')
    exit()


def model_evaluation(mask_win=0,
                     mask_start_idx=0,
                     snr=10, set='', is_auto=False):
    # Fixme
    net = Net.DCNN_MutilChannel(in_channel=1, out_channel=10)
    config, load_files = dcnn_config()
    load_files = config[set]
    config['resultPath'] = os.path.join(config['resultPath'], r'{}_official'.format(net.log))


    """ 测试网络 """
    # result = train.test_model(dataLoader=testloader, pretrain=True,
    #                           file='checkpoint_2022-10-10 17.56.05_epo[1724].model'
    #                           )
    # print('test model:', result)

    print(""" ============================================================ """)
    if not is_auto:
        mask_win = 0
        mask_start_idx = 0
        snr = 10
    train_dataset, test_dataset = DataSet_CRWU_MultiFile().get_train_and_test_set_ordered_enhance(
        root=config['train_loader']['dataset']['path'],
        load_files=load_files,
        train_test_rate=0.5, repeat_win=0.85, window=1024,
        mask_win=int(1024 * mask_win),
        mask_start_idx=int(1024 * mask_start_idx),
        snr=snr)  # cate='test',  #  int(2048 * 0.6)  DataSet_CRWU_Signal_Process
    # train_dataset, test_dataset = DataSet_CRWU_MultiFile().get_train_and_test_set_ordered(
    #                                         config['train_loader']['dataset']['path'], load_files,
    #                                         train_test_rate=0.5, repeat_win=0.85, window=2048)
    print('orignal size={}  new test_dataset size={}'.format(len(train_dataset), len(test_dataset)))

    testloader = data.DataLoader(dataset=test_dataset, batch_size=216,
                                 shuffle=False, drop_last=False, )
    print('new test_dataloader size={}'.format(len(testloader)))

    train = Train.Train(net=net, train_dataLoader=None, test_dataLoader=testloader,
                        config=config, write_csv=True, save_log=True, tensorboard_mode='loss-acc',
                        new_thread=True)

    result = train.test_model_visualization(dataLoader=testloader, pretrain=True,
                                            file='checkpoint_2022-10-19 17.38.27_epo[333]_best.model'
                                            )
    print('test model:', result)

    from matplotlib import pyplot as plt
    plt.title(f'mask size={mask_win} mask idx={mask_start_idx} SNR={snr}')
    if not is_auto:
        plt.show()
    else:
        path = config['resultPath'] + '\\' + r'matrix img'
        file = 'matrix mask{} maskidx{} snr{}.png'.format(mask_win, mask_start_idx, snr)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, file)
        print('save fig at:', path)
        plt.savefig(path)
    return result['acc']
    # plt.show()


def run():
    """ 论文官方训练方式：数据无处理，不加噪声，不翻倍 """
    official_train()

    """ 数据处理，加噪声 """
    # designed_train()

    """ 模型性能测试 """
    # model_evaluation()
    exit()

    mask_size = [0, 0.2, 0.4, 0.6, 0.8]
    mask_idx = [0, 0.2, 0.4, 0.6, 0.8]
    snr = [-4, -2, 0, 2, 4, 6, 8, 10]

    model = 'checkpoint_2022-11-02 14.24.30_epo[330]_best.model'
    eval_set = 'crwu_set2'
    train_way = dict(
        a=dict(network='dcnn', dataset='crwu1_2', train_size=2048, evalute_szie=2048),
        b=dict(mask='0', mask_idx='0', SNR='random', eval_set=eval_set),  # random  #  0
    )
    dd = dict(save_file=r'I:\python\01-work\result\02-pro\dcnn_record.xlsx',
              model_path=os.path.join(config['resultPath'], 'DCNN_official', 'model', model),
              train_way=train_way,
              start_row=1 + 23 * 2, sheet=0,
              )
    report = Make_Report(**dd)
    for ms in mask_size:
        for mi in mask_idx:
            print('running:', ms, '---', mi, )
            for sn in snr:
                acc = model_evaluation(mask_win=ms, mask_start_idx=mi, snr=sn,set=eval_set , is_auto=True)
                report.add_data(mask_size=ms, mask_idx=mi, snr=sn, acc=acc)

            if ms + mi >= 1 or ms == 0: break

    report.record_data_in_excel()


if __name__ == '__main__':
    run()
