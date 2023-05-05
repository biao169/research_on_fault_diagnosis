import random
import torch
import torch.utils
from torch.utils import data
import numpy as np
import os

from dataset.dataset_CRWU import DataSet_CRWU_MultiFile
from dataset.dataset_MFPT import DataSet_MFPT_MultiFile
from networks.sota_nets.n03_WPD_MSCNN.wpd_mscnn import WPD_MSCNN as Net
from networks.sota_nets.n03_WPD_MSCNN.train_wpdmscnn import Train
from utils_tool.log_utils import YamlHandler, delete_all_log_record_file

from report_make.excel_report import Make_Report

"""
    本部分程序只用于训练 复现 1DCNN-LSTM
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

datasetName = 'MFPT'
train_set = 'mfpt_set1'
num_class = len(config['data_resource'][datasetName][train_set])
window = 2048


def get_config(config=config, set='mfpt_set1'):
    root = config['data_resource'][datasetName]['path']
    config['train_loader']['dataset']['path'] = root
    config['test_loader']['dataset']['path'] = root
    load_files = config['data_resource'][datasetName][set]  ## 'mfpt_set1'   crwu_set1_2   crwu_set1
    """ official parameter """
    config['optimizer'] = {}
    # config['scheduler'] = {}
    config['optimizer']['Adam'] = dict(lr=0.001, betas=[0.9, 0.999])  # , weight_decay=0.0001
    # config['scheduler']['StepLR'] = dict(step_size=50, gamma=0.1, last_epoch=-1)
    config['train_loader']['dataset']['miniBatch'] = 128
    config['test_loader']['dataset']['miniBatch'] = 128
    return config.copy(), load_files


def get_config2(config=config, set='mfpt_set1'):
    """ Only for evaluation in another condition or fault """
    load_files = config['data_resource'][datasetName][set]
    config, _ = get_config()
    return config, load_files


def official_train():
    # Fixme
    net = Net(in_channel=1, out_channel=num_class)
    config, load_files = get_config()
    config['resultPath'] = os.path.join(config['resultPath'], r'{}_official'.format(net.log))

    # train_dataset, test_dataset = DataSet_MFPT_MultiFile().get_train_and_test_set_ordered(
    #     config['train_loader']['dataset']['path'], load_files,
    #     train_test_rate=0.7, repeat_win=0.85, window=1024)
    train_dataset, test_dataset = DataSet_MFPT_MultiFile().get_train_and_test_set_ordered_enhance(
        root=config['train_loader']['dataset']['path'],
        load_files=load_files,
        train_test_rate=0.7, repeat_win=0.85, window=window,
        mask_win=int(window * 0.2),
        mask_start_idx=None,
        snr=None)  # cate='test',  #  int(2048 * 0.6)  DataSet_CRWU_Signal_Process
    print(f'train_dataset len = {len(train_dataset)},\t test_dataset len = {len(test_dataset)}')

    trainloader = data.DataLoader(dataset=train_dataset, batch_size=config['train_loader']['dataset']['miniBatch'],
                                  shuffle=config['train_loader']['shuffle'], drop_last=False, )
    testloader = data.DataLoader(dataset=test_dataset, batch_size=config['test_loader']['dataset']['miniBatch'],
                                 shuffle=config['test_loader']['shuffle'], drop_last=False, )

    """ 删除日志文件 """
    delete_all_log_record_file(root=config['resultPath'], idx=[1], tm=None, rm_config=True, rm_csv=True,
                               rm_model=True, rm_tensor=True)
    config['epochNum'] = 800
    train = Train(net=net, train_dataLoader=trainloader, test_dataLoader=testloader,
                  config=config, write_csv=True, save_log=True, tensorboard_mode='loss-acc',
                  new_thread=True)
    train.training_mode(pretrain=False, datasetName=datasetName)
    exit()
    pass


def designed_train():
    # Fixme
    net = Net(in_channel=1, out_channel=num_class)

    config, load_files = get_config()
    config['resultPath'] = os.path.join(config['resultPath'], r'{}_designed'.format(net.log))

    # train_dataset, test_dataset = DataSet_MFPT_MultiFile().get_train_and_test_set_ordered_enhance(
    #                             config['train_loader']['dataset']['path'], load_files,
    #                             train_test_rate=0.5, repeat_win=0.85, window=2048)
    train_dataset, test_dataset = DataSet_MFPT_MultiFile().get_train_and_test_set_ordered_enhance(
        root=config['train_loader']['dataset']['path'],
        load_files=load_files,
        train_test_rate=0.5, repeat_win=0.85, window=window,
        mask_win=int(window * 0.2),
        mask_start_idx=None,
        snr=None)  # cate='test',  #  int(2048 * 0.6)  DataSet_CRWU_Signal_Process
    config['train way'] = dict(mask_win=0.8, mask_start_idx=None, snr=None)
    print(f'train_dataset len = {len(train_dataset)},\t test_dataset len = {len(test_dataset)}')

    trainloader = data.DataLoader(dataset=train_dataset, batch_size=config['train_loader']['dataset']['miniBatch'],
                                  shuffle=config['train_loader']['shuffle'], drop_last=False, )
    testloader = data.DataLoader(dataset=test_dataset, batch_size=config['test_loader']['dataset']['miniBatch'],
                                 shuffle=config['test_loader']['shuffle'], drop_last=False, )

    """ 删除日志文件 """
    delete_all_log_record_file(root=config['resultPath'], idx=[], tm=None, rm_config=True, rm_csv=True,
                               rm_model=True, rm_tensor=True)
    config['epochNum'] = 2000
    train = Train(net=net, train_dataLoader=trainloader, test_dataLoader=testloader,
                  config=config, write_csv=True, save_log=True, tensorboard_mode='loss-acc',
                  new_thread=True)
    train.training_mode(pretrain=False, datasetName=datasetName)
    exit()


def model_evaluation(mask_win=0,
                     mask_start_idx=0,
                     snr=10, model=None, set='', is_auto=False):
    # Fixme
    net = Net(in_channel=1, out_channel=num_class)
    config, load_files = get_config()
    if not (set is None or set == ''): config, load_files = get_config(set=set)
    config['resultPath'] = os.path.join(config['resultPath'], r'{}_official'.format(net.log))

    """ 测试网络 """
    print(f" ============================== is_auto: {is_auto} ============================== ")
    if not is_auto:
        mask_win = 0
        mask_start_idx = 0
        snr = 10
        model = ''

    train_dataset, test_dataset = DataSet_MFPT_MultiFile().get_train_and_test_set_ordered_enhance(
        root=config['train_loader']['dataset']['path'],
        load_files=load_files,
        train_test_rate=0.7, repeat_win=0.85, window=window,
        mask_win=int(window * mask_win),
        mask_start_idx=int(window * mask_start_idx),
        snr=snr)  # cate='test',  #  int(2048 * 0.6)  DataSet_CRWU_Signal_Process
    # train_dataset, test_dataset = DataSet_MFPT_MultiFile().get_train_and_test_set_ordered(
    #                                         config['train_loader']['dataset']['path'], load_files,
    #                                         train_test_rate=0.5, repeat_win=0.85, window=1024)
    print('orignal size={}  new test_dataset size={}'.format(len(train_dataset), len(test_dataset)))

    testloader = data.DataLoader(dataset=test_dataset, batch_size=2048 * 10,
                                 shuffle=False, drop_last=False, )
    print('new test_dataloader size={}'.format(len(testloader)))

    train = Train(net=net, train_dataLoader=None, test_dataLoader=testloader,
                  config=config, write_csv=True, save_log=True, tensorboard_mode='loss-acc',
                  new_thread=True)

    result = train.test_model_visualization(dataLoader=testloader, pretrain=True, datasetName=datasetName,
                                            file=model
                                            )
    print('test model:', result)

    from matplotlib import pyplot as plt
    plt.title(f'mask size={mask_win}  mask idx={mask_start_idx}  SNR={snr}')
    plt.draw()
    if not is_auto:
        plt.show()
    else:
        plt.pause(0.1)
        path = config['resultPath'] + '\\' + r'matrix img'
        file = 'matrix mask{} maskidx{} snr{}.png'.format(mask_win, mask_start_idx, snr)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, file)
        print('save fig at:', path)
        plt.savefig(path)

    return result['acc']
    # plt.show()


def t_sne_vishow(mask_win=0, mask_start_idx=0, snr=0, model='', set='', is_auto=False, isave=False):
    from utils_tool.t_SNE import t_SNE_Visualization as T_SNE
    # Fixme
    net = Net.LSTM_1DC(in_channel=1, out_channel=num_class)
    config, load_files = get_config()
    if not (set is None or set == ''): config, load_files = get_config(set=set)
    config['resultPath'] = os.path.join(config['resultPath'], r'{}_official'.format(net.work_name))
    print(net)
    """ 测试网络 """
    tsne = T_SNE()
    tsne.set_hook(net.fc, 2)

    print(""" ============================================================ """)
    if not is_auto:
        mask_win = 0.2
        mask_start_idx = None
        snr = None
        model = 'checkpoint_2022-11-06 20.47.47_epo[494]_best.model'
    train_dataset, test_dataset = DataSet_MFPT_MultiFile().get_train_and_test_set_ordered_enhance(
        # train_dataset, test_dataset = DataSet_KAT_MultiFile().get_train_and_test_set_ordered_enhance(
        # train_dataset, test_dataset=DataSet_DNU_MultiFile().get_train_and_test_set_ordered_enhance(
        root=config['train_loader']['dataset']['path'],
        load_files=load_files,
        train_test_rate=0.5, repeat_win=0.85, window=window,
        mask_win=int(window * mask_win),
        mask_start_idx=int(window * mask_start_idx) if mask_start_idx else None,  #
        snr=snr)  # cate='test',  #  int(2048 * 0.6)  DataSet_CRWU_Signal_Process
    print('orignal size={}  new test_dataset size={}'.format(len(train_dataset), len(test_dataset)))

    testloader = data.DataLoader(dataset=test_dataset, batch_size=512,
                                 shuffle=False, drop_last=False, )
    print('new test_dataloader size={}'.format(len(testloader)))

    train = Train(net=net, train_dataLoader=None, test_dataLoader=testloader,
                  config=config, write_csv=True, save_log=True, tensorboard_mode='loss-acc',
                  new_thread=True)
    net, epoch = train.load_model_weight_auto(net, file=model)
    for i, dat in enumerate(testloader):
        ''' Design program '''
        if datasetName.lower() == 'crwu':
            label, _, _, de, fe, ba = dat[:6]
        elif datasetName.lower() == 'mfpt':
            label, _, _, de = dat[:4]
        de, fe, label = de.to(train.device), fe.to(train.device), label.to(train.device)
        output = net([de, fe])
        tsne.add_label(label)

    tsne.t_sne_plot()
    from matplotlib import pyplot as plt
    plt.title(f'net={net.log}  mask size={mask_win}  mask idx={mask_start_idx}  SNR={snr}')
    plt.draw()
    plt.pause(0.1)

    if isave:
        path = config['resultPath'] + '\\' + r'matrix img'
        file = 't-sne {} matrix mask{} maskidx{} snr{}.png'.format(net.log, mask_win, mask_start_idx, snr)
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, file)
        print('save fig at:', path)
        plt.savefig(path)

    plt.show()
    exit()
    pass


def run():
    """ 论文官方训练方式：数据无处理，不加噪声，不翻倍 """
    # official_train()

    """ 数据处理，加噪声 """
    # designed_train()

    """ 模型性能测试 """
    # model_evaluation()
    # t_sne_vishow(isave=True)
    # exit()

    mask_size = [0, 0.2, 0.4, 0.6, 0.8]  # #
    mask_idx = [0, 0.2, 0.4, 0.6, 0.8]  #
    snr = [-4, -2, 0, 2, 4, 6, 8, 10]  #

    model = 'checkpoint_2023-04-06 11.26.50_epo[598]_best.model'
    eval_set = train_set  # 'mfpt_set1'  # 'crwu_set1_2'
    train_way = dict(
        a=dict(network='wdp-mscnn', dataset=train_set, train_size=window, evalute_szie=window),
        b=dict(mask='random', mask_idx='random', SNR='random', eval_set=eval_set),  # random  0
    )
    dd = dict(save_file=os.path.join(config['resultPath'], 'wpd_mscnn_record.xlsx'),
              model_path=os.path.join(config['resultPath'], 'WPD_MSCNN_official', 'model', model),
              train_way=train_way,
              start_row=1 + 23 * 0, sheet=0,
              )

    report = Make_Report(**dd)
    for ms in mask_size:
        for mi in mask_idx:
            print('running:', ms, '---', mi, )
            for sn in snr:
                acc = model_evaluation(mask_win=ms, mask_start_idx=mi, snr=sn, model=model, set=eval_set, is_auto=True)
                report.add_data(mask_size=ms, mask_idx=mi, snr=sn, acc=acc)

            if ms + mi >= 1 or ms == 0: break

    report.record_data_in_excel()


if __name__ == '__main__':
    run()
